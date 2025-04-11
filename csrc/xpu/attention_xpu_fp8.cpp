// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <ext/intel/esimd.hpp>
#include "kv.h"

// clang-format on
#include <float.h>
#include <torch/extension.h>
#include <stdexcept>
#include "utils.h"
#include "xpu_types.h"
// #include "dtype_bfloat16.dp.hpp"
#include "dtype_float16.h"
#include "dtype_float32.h"
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
#include <c10/xpu/XPUStream.h>
#endif

#include <functional>
// #include <ipex.h>

using namespace sycl::ext::intel::esimd;
using AT = at::ScalarType;

template <typename IT, const int VS, const int HD>
void gqa_1_kernel_fp8(
    const void* query,  // [num_seqs, num_heads, head_size]
    const void* key,    // [num_blocks, num_kv_heads, head_size, block_size]
    const void* value,  // [num_blocks, num_kv_heads, head_size, block_size]
    const void* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const void* context_lens,  // [num_seqs]
    void* o_a_s, void* o_accs, const int64_t query_bsz_stride,
    const int64_t query_head_stride, const int64_t kv_token_stride,
    const int64_t kv_head_stride, const int64_t kv_block_stride,
    const int64_t block_table_stride_batch, const int64_t o_a_s_bsz_stride,
    const int64_t o_a_s_head_stride, const int64_t o_accs_bsz_stride,
    const int64_t o_accs_head_stride, const float scale, const int block_size,
    const int bsz, const int num_heads, const int num_kv_heads,
    const int block_num, const at::Device& device) {
  const int group_size = num_heads / num_kv_heads;
  const int sub_rows = VS / group_size;
  const int rem_rows = VS % group_size;

  const float attn_scale = scale;

  sycl::range<3> global_size(bsz, num_heads, block_num);
  sycl::range<3> local_size(1, group_size, 1);

  auto cgf = [&](sycl::handler& handle) {
    handle.parallel_for(
        sycl::nd_range<3>(global_size, local_size),
        [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          slm_init<VS * HD * sizeof(IT)>();

          const int bsz_idx = item.get_global_id(0);
          const int head_idx = item.get_global_id(1);
          const int kv_head_idx = item.get_group(1);
          const int tid = item.get_local_id(1);
          const int vid = item.get_global_id(2);

          const IT* query_head = (const IT*)query + bsz_idx * query_bsz_stride +
                                 head_idx * query_head_stride;

          IT* o_accs_head = (IT*)o_accs + bsz_idx * o_accs_bsz_stride +
                            head_idx * o_accs_head_stride;
          float* o_a_s_head = (float*)o_a_s + bsz_idx * o_a_s_bsz_stride +
                              head_idx * o_a_s_head_stride;

          const int* block_tables_ptr = (const int*)block_tables;
          const int* block_table =
              block_tables_ptr + bsz_idx * block_table_stride_batch;

          const int* context_lens_ptr = (const int*)context_lens;
          const int context_length = context_lens_ptr[bsz_idx];

          simd<IT, HD> query_row = block_load<IT, HD>(query_head) * attn_scale;

          // copy k_cache to slm
          int start_row =
              std::min(vid * VS + tid * sub_rows + std::min(tid, rem_rows),
                       context_length);
          int end_row =
              std::min(start_row + sub_rows + (tid < rem_rows), context_length);
          for (int r = start_row; r < end_row; ++r) {
            int which_block = r / block_size;
            int which_slot = r % block_size;
            int physical_block_number = block_table[which_block];

            // Load elements in uint8_t
            const uint8_t* key_head =
                (const uint8_t*)key + physical_block_number * kv_token_stride +
                kv_head_idx * kv_head_stride + which_slot * kv_block_stride;

            simd<uint8_t, HD> key_row = block_load<uint8_t, HD>(key_head);
            simd<IT, HD> key_dequantized = dequantize_key_row<HD>(key_row);
            slm_block_store<IT, HD>((r - vid * VS) * HD * sizeof(IT), key_dequantized);
          }
          barrier();

          simd<float, VS> attns = -sycl::detail::max_v<float>();
          int row_num =
              (vid + 1) * VS > context_length ? context_length % VS : VS;
          // q @ k
          for (int r = 0; r < row_num; ++r) {
            simd<IT, HD> key_row = slm_block_load<IT, HD>(r * HD * sizeof(IT));
            float attn = sycl::ext::intel::esimd::detail::sum<float, IT, HD>(
                query_row * key_row);
            attns[r] = attn;
          }

          float max_attn = hmax<float, float, VS>(attns);
          const simd<IT, VS> attn_exp = exp(attns - max_attn);
          barrier();

          // copy v_cache to slm
          for (int r = start_row; r < end_row; ++r) {
            int which_block = r / block_size;
            int which_slot = r % block_size;
            int physical_block_number = block_table[which_block];

            const uint8_t* value_head =
                (const uint8_t*)value + physical_block_number * kv_token_stride +
                kv_head_idx * kv_head_stride + which_slot * kv_block_stride;

            simd<uint8_t, HD> value_row = block_load<uint8_t, HD>(value_head);
            simd<IT, HD> value_dequantized = dequantize_value_row<HD>(value_row);
            slm_block_store<IT, HD>((r - vid * VS) * HD * sizeof(IT),
                                    value_dequantized);
          }
          barrier();

          // attn @ v
          simd<IT, HD> accs = 0;
          for (int r = 0; r < row_num; ++r) {
            simd<IT, HD> value_row =
                slm_block_load<IT, HD>(r * HD * sizeof(IT));
            accs = accs + value_row * attn_exp[r];
          }

          float softmax =
              sycl::ext::intel::esimd::detail::sum<float, float, VS>(attn_exp);

          block_store<IT, HD>(o_accs_head + vid * HD, accs);
          block_store<float, 1>(o_a_s_head + vid * 2, max_attn);
          block_store<float, 1>(o_a_s_head + vid * 2 + 1, softmax);
        });
  };

  utils::submit_kernel(cgf, device, "gqa kernel 1/2");
}

template <typename IT, const int GS, const int HD>
void gqa_2_kernel_fp8(void* o_a_s, void* o_accs, void* output,
                  const void* context_lens,  // [num_seqs]
                  const int64_t o_a_s_bsz_stride,
                  const int64_t o_a_s_head_stride,
                  const int64_t o_accs_bsz_stride,
                  const int64_t o_accs_head_stride,
                  const int64_t output_bsz_stride,
                  const int64_t output_head_stride, const int bsz,
                  const int num_heads, const int row_block_num,
                  const at::Device& device) {
  constexpr int SUB_HD = 8;
  static_assert(HD % SUB_HD == 0);
  static_assert(HD / SUB_HD <= GS);

  const int sub_rows = row_block_num / GS;
  const int rem_rows = row_block_num % GS;

  constexpr int accs_slm_offset = 0;
  constexpr int attn_slm_offset = GS * HD * sizeof(float);
  constexpr int softmax_slm_offset = attn_slm_offset + GS * sizeof(float);

  sycl::range<3> global_size(bsz, num_heads, GS);
  sycl::range<3> local_size(1, 1, GS);

  auto cgf = [&](sycl::handler& handle) {
    handle.parallel_for(
        sycl::nd_range<3>(global_size, local_size),
        [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          slm_init<GS * HD * sizeof(float) + GS * 2 * sizeof(float)>();

          const int bsz_idx = item.get_global_id(0);
          const int head_idx = item.get_global_id(1);
          const int tid = item.get_global_id(2);

          const int* context_lens_ptr = (const int*)context_lens;
          const int context_length = context_lens_ptr[bsz_idx];
          constexpr int VS = 32;
          const int cur_row_block_num = (context_length + VS - 1) / VS;
          const int cur_sub_rows = cur_row_block_num / GS;
          const int cur_rem_rows = cur_row_block_num % GS;

          const float* o_a_s_head = (const float*)o_a_s +
                                    bsz_idx * o_a_s_bsz_stride +
                                    head_idx * o_a_s_head_stride;
          const IT* o_accs_head = (const IT*)o_accs +
                                  bsz_idx * o_accs_bsz_stride +
                                  head_idx * o_accs_head_stride;
          IT* output_head = (IT*)output + bsz_idx * output_bsz_stride +
                            head_idx * output_head_stride;

          int start_row =
              std::min(tid * cur_sub_rows + std::min(tid, cur_rem_rows),
                       cur_row_block_num);
          int end_row =
              std::min(start_row + cur_sub_rows + (tid < cur_rem_rows),
                       cur_row_block_num);

          float max_attn = -sycl::detail::max_v<float>();
          float softmax = 0;
          simd<float, HD> accs = 0;
          for (int r = start_row; r < end_row; ++r) {
            float sub_attn = o_a_s_head[2 * r];
            float sub_softmax = o_a_s_head[2 * r + 1];
            simd<float, HD> sub_accs = block_load<IT, HD>(o_accs_head + r * HD);
            float new_max_attn = std::max(max_attn, sub_attn);
            float exp1 = exp(max_attn - new_max_attn);
            float exp2 = exp(sub_attn - new_max_attn);
            accs = accs * exp1 + sub_accs * exp2;
            softmax = softmax * exp1 + sub_softmax * exp2;
            max_attn = new_max_attn;
          }

          slm_block_store<float, HD>(accs_slm_offset + tid * HD * sizeof(float),
                                     accs);
          slm_block_store<float, 1>(attn_slm_offset + tid * sizeof(float),
                                    max_attn);
          slm_block_store<float, 1>(softmax_slm_offset + tid * sizeof(float),
                                    softmax);
          barrier();

          if (tid < HD / SUB_HD) {
            simd<float, GS> max_attns =
                slm_block_load<float, GS>(attn_slm_offset);
            const simd<float, GS> scales =
                exp(max_attns - hmax<float, float, GS>(max_attns));
            simd<float, GS> softmaxs =
                slm_block_load<float, GS>(softmax_slm_offset);
            float softmax_sum =
                sycl::ext::intel::esimd::detail::sum<float, float, GS>(
                    softmaxs * scales);

            simd<float, SUB_HD> result = 0;
#pragma unroll
            for (int r = 0; r < GS; ++r) {
              simd<float, SUB_HD> sub_accs = slm_block_load<float, SUB_HD>(
                  accs_slm_offset + (r * HD + tid * SUB_HD) * sizeof(float));
              result = result + sub_accs * scales[r];
            }
            result = result / softmax_sum;
            block_store<IT, SUB_HD>(output_head + tid * SUB_HD, result);
          }
        });
  };

  utils::submit_kernel(cgf, device, "gqa kernel 2/2");
}

template <const int VS, const int GS, const int HD>
auto dispatch_gqa_kernel_fp8(AT it) {
  switch (it) {
    case AT::Float:
      return std::make_tuple(gqa_1_kernel_fp8<float, VS, HD>,
                             gqa_2_kernel_fp8<float, GS, HD>);
    case AT::Half:
      return std::make_tuple(gqa_1_kernel_fp8<fp16, VS, HD>,
                             gqa_2_kernel_fp8<fp16, GS, HD>);
    default:
      throw std::runtime_error(
          "unsupported dtype, only fp32 and fp16 are supported");
  }
}

void paged_attention_gqa_fp8(torch::Tensor output, torch::Tensor query,
                         torch::Tensor key_cache, torch::Tensor value_cache,
                         int64_t bsz, int64_t num_heads, int64_t num_kv_heads,
                         float scale, torch::Tensor& block_tables,
                         torch::Tensor& context_lens, int block_size,
                         int64_t head_dim, int max_seq_len) {
  constexpr int VS = 32;
  constexpr int GS = 32;

  const int row_block_num = (max_seq_len + VS - 1) / VS;
  auto o_a_s =
      torch::empty({bsz, num_heads, 1, row_block_num * 2},
                   torch::device(query.device()).dtype(torch::kFloat32));
  auto o_accs =
      torch::empty({bsz, num_heads, 1, row_block_num * head_dim},
                   torch::device(query.device()).dtype(query.dtype()));

  auto [func1, func2] = [&]() {
    switch (head_dim) {
      case 128:
        return dispatch_gqa_kernel_fp8<VS, GS, 128>(query.scalar_type());
      case 96:
        return dispatch_gqa_kernel_fp8<VS, GS, 96>(query.scalar_type());
      case 80:
        return dispatch_gqa_kernel_fp8<VS, GS, 80>(query.scalar_type());
      case 64:
        return dispatch_gqa_kernel_fp8<VS, GS, 64>(query.scalar_type());
      default:
        throw std::runtime_error(
            "unsupported head_dim, only 128, 96, 80 and 64 are supported");
    }
  }();

  func1(query.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
        block_tables.data_ptr(), context_lens.data_ptr(), o_a_s.data_ptr(),
        o_accs.data_ptr(), query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        block_tables.stride(0), o_a_s.stride(0), o_a_s.stride(1),
        o_accs.stride(0), o_accs.stride(1), scale, block_size, bsz, num_heads,
        num_kv_heads, row_block_num, query.device());

  func2(o_a_s.data_ptr(), o_accs.data_ptr(), output.data_ptr(),
        context_lens.data_ptr(), o_a_s.stride(0), o_a_s.stride(1),
        o_accs.stride(0), o_accs.stride(1), output.stride(0), output.stride(1),
        bsz, num_heads, row_block_num, query.device());
}
