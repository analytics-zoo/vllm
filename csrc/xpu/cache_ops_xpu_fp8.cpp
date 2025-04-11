// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <ext/intel/esimd.hpp>
// clang-format on
#include "xpu_types.h"

#include <torch/extension.h>
#include "utils.h"
#include "kv.h"

using fp16 = sycl::half;
using namespace sycl::ext::intel::esimd;

// scalar_t is key.scalar_type() -> half
template <typename scalar_t, const int HD>
void reshape_and_cache_ipexllm_kernel_fp8(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    uint8_t * __restrict__ key_cache,  // [num_blocks, num_kv_heads, block_size,
                                       // head_size]
    uint8_t * __restrict__ value_cache,        // [num_blocks, num_kv_heads,
                                               // block_size, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, 
    const int key_head_stride, const int value_head_stride,
    const int num_heads,
    const int head_size, const int block_size, const int x,
    const sycl::nd_item<3>& item_ct1) {
  
  //                      New Implementation                      //
  const size_t token_idx = item_ct1.get_global_id(0);
  const size_t head_idx = item_ct1.get_global_id(1);
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  // The thread is responsible for the HD elements within key/value
  const scalar_t * key_head = key + token_idx * key_stride + head_idx * key_head_stride;

  const scalar_t * value_head = value + token_idx * value_stride + head_idx * value_head_stride;

  uint8_t * key_output_head = key_cache + block_idx * num_heads * head_size * block_size + 
      head_idx * head_size * block_size + block_offset * head_size;
  uint8_t * value_output_head = value_cache + block_idx * num_heads * head_size * block_size + 
      head_idx * head_size * block_size + block_offset * head_size;
  
  simd<fp16, HD> key_row = block_load<scalar_t, HD>(key_head);
  simd<uint8_t, HD> key_result = quantize_key_row<HD>(key_row);
  block_store<uint8_t, HD>(key_output_head, key_result);

  simd<fp16, HD> value_row = block_load<scalar_t, HD>(value_head);
  simd<uint8_t, HD> value_result = quantize_value_row<HD>(value_row);
  block_store<uint8_t, HD>(value_output_head, value_result);
}


template <typename scalar_t, const int HD>
void call_reshape_and_cache_ipexllm_kernel_fp8(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int num_tokens,
    const int key_stride, const int value_stride, 
    const int key_head_stride, const int value_head_stride,
    const int num_heads,
    const int head_size, const int block_size, const int x) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  sycl::range<3> grid(num_tokens, num_heads, 1);
  sycl::range<3> block(1, 1, 1);
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) SYCL_ESIMD_KERNEL {
          reshape_and_cache_ipexllm_kernel_fp8<sycl_t, HD>(
              (const sycl_t* __restrict__)key,
              (const sycl_t* __restrict__)value,
              (uint8_t* __restrict__)key_cache,
              (uint8_t* __restrict__)value_cache, slot_mapping, key_stride,
              value_stride, key_head_stride, value_head_stride,
              num_heads, head_size, block_size, x, item_ct1);
        });
  });
}

void reshape_and_cache_ipexllm_fp8(torch::Tensor& key, torch::Tensor& value,
                               torch::Tensor& key_cache,
                               torch::Tensor& value_cache,
                               torch::Tensor& slot_mapping,
                               const std::string& kv_cache_dtype,
                               const float kv_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(2);
  // int x = key_cache.size(4);
  int x = 1;

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  int key_head_stride = key.stride(1);
  int value_head_stride = value.stride(1);

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "call_reshape_and_cache_ipexllm_kernel_fp8", [&] {
        call_reshape_and_cache_ipexllm_kernel_fp8<scalar_t, 128>(
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(), num_tokens, key_stride,
            value_stride, key_head_stride, value_head_stride,
            num_heads, head_size, block_size, x);
      });
}



