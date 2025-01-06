#include <torch/extension.h>
#include <ext/intel/esimd.hpp>

#include "kv.h"
#include "device.h"
#include "utils.h"
#include "memory.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using ST = at::ScalarType;

using namespace torch::indexing;
using namespace sycl::ext::intel::esimd;

template<typename IT, const int N>
ESIMD_INLINE simd<IT, N> cblock_load(const IT* addr, bool condition) {
    if (condition) {
        return block_load<IT, N>(addr);
    } else {
        return 0;
    }
}

template<typename IT, const int Depth, const int ExecuteSize, const int HD>
void reshape_key_cache_kernel(
    const void * key,
    const void* block_tables, // [num_seqs, max_num_blocks_per_seq]
    const void* context_lengths,
    const int64_t key_token_stride,
    const int64_t key_head_stride,
    const int64_t block_table_stride_batch,
    const int bsz,
    const int num_kv_heads,
    const int padding_context_length,
    const torch::Device & device
) {
    sycl::range<3> global_size(bsz, num_kv_heads, padding_context_length / ExecuteSize);
    sycl::range<3> local_size(1, 1, 1);
    auto cgf = [=](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int bsz_idx = item.get_global_id(0);
                const int head_idx = item.get_global_id(1);
                const int block_idx = item.get_global_id(2);

                const int* block_tables_ptr = (const int*)block_tables;
                const int* block_table =
                    block_tables_ptr + bsz_idx * block_table_stride_batch;
                
                const int32_t* context_len_ptr = (const int32_t*)context_lens;
                const int32_t context_length = context_len_ptr[bsz_idx];

                int physical_block_number = block_table[block_idx];

                const IT * key_block = (const IT *)key + physical_block_number * key_token_stride
                                                       + head_idx * key_head_stride;

                simd<fp16, HD * ExecuteSize> new_key_rows;
                #pragma unroll
                for (int i = 0; i < ExecuteSize; ++i) {
                    simd<fp16, HD> key_row =
                        cblock_load<IT, HD>(key_block + i * HD, block_idx * ExecuteSize + i < context_length);
                    new_key_rows.template bit_cast_view<float>().template select<HD / 2, ExecuteSize>(i) =
                        key_row.template bit_cast_view<float>();
                }

                block_store<fp16, ExecuteSize * HD>(key_block, new_key_rows);
            }
        );
    };
    utils::submit_kernel(cgf, device, "reshape key cache");
}

template<typename IT, const int Depth, const int ExecuteSize, const int HD, const int KVS>
void reshape_value_cache_kernel(
    const void * value,
    const void* block_tables,
    const void* context_lengths,
    const int64_t value_token_stride,
    const int64_t value_head_stride,
    const int64_t new_value_token_stride,
    const int64_t new_value_head_stride,
    const int64_t block_table_stride_batch,
    const int bsz,
    const int num_kv_heads,
    const int padding_context_length,
    const torch::Device & device
) {
    sycl::range<3> global_size(bsz, num_kv_heads, padding_context_length / ExecuteSize); // TODO: check this modification and refine block_size setting.
    sycl::range<3> local_size(1, 1, 1);
    auto cgf = [=](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<ExecuteSize * HD * sizeof(fp16)>();

                const int bsz_idx = item.get_global_id(0);
                const int head_idx = item.get_global_id(1);
                const int block_idx = item.get_global_id(2);

                const int* block_tables_ptr = (const int*)block_tables;
                const int* block_table =
                    block_tables_ptr + bsz_idx * block_table_stride_batch;
                
                const int32_t* context_len_ptr = (const int32_t*)context_lens;
                const int32_t context_length = context_len_ptr[bsz_idx];

                int physical_block_number = block_table[block_idx];

                const IT * value_block = (const IT *)value + physical_block_number * value_token_stride
                                                           + head_idx * value_head_stride;

                #pragma unroll
                for (int i = 0; i < ExecuteSize; i += 2) {
                    simd<fp16, 2 * HD> merged_value_row;
                    merged_value_row.template select<HD, 2>(0) =
                        cblock_load<IT, HD>(value_block + i * HD, block_idx * KVS + i < context_length);
                    merged_value_row.template select<HD, 2>(1) =
                        cblock_load<IT, HD>(value_block + (i + 1) * HD, block_idx * KVS + i + 1 < context_length);

                    #pragma unroll
                    for (int j = 0; j < HD / ExecuteSize; ++j) {
                        slm_block_store<fp16, 2 * ExecuteSize>(
                            j * ExecuteSize * KVS + i / 2 * 2 * ExecuteSize,
                            merged_value_row.template select<2 * ExecuteSize, 1>(j * 2 * ExecuteSize)
                        );
                    }
                }

                simd<IT, HD> new_value_rows = slm_block_load<fp16, ExecuteSize * HD>(0);
                block_store<fp16, ExecuteSize * HD>(value_block, new_value_rows);
            }
        );
    };
    utils::submit_kernel(cgf, device, "reshape value cache");
}

// force `block_num`'s register offset to workaround a bug of compiler
static ESIMD_PRIVATE ESIMD_REGISTER(2560) simd<int, 1> block_num;

template<typename IT, typename AT, const int HD, const int GS, const int KVS,
         const int RepeatCount, const int Depth, const int ExecuteSize>
void chunked_prefill_xmx_kernel(
    const void * query,
    const void * key,
    const void * value,
    const void* block_tables, // [num_seqs, max_num_blocks_per_seq]
    const void* query_start_loc,
    const void* seq_lens,
    const void* context_lens,
    void * tmp_output,
    void * output,
    const int64_t query_token_stride,
    const int64_t query_head_stride,
    const int64_t key_token_stride,
    const int64_t key_head_stride,
    const int64_t value_token_stride,
    const int64_t value_head_stride,
    const int64_t tmp_output_bsz_stride, // TODO(xiangyu): check this
    const int64_t tmp_output_head_stride,
    const int64_t output_token_stride,
    const int64_t output_head_stride,
    const int64_t block_table_stride_batch,
    const float attn_scale,
    const int block_size,
    const int bsz,
    const int num_heads,
    const int num_kv_heads,
    const int max_input_length,
    const torch::Device & device
) {
    constexpr int QS = RepeatCount * GS;
    constexpr int RS = KVS / GS; // 需要rs可以整除block_size

    static_assert(Depth % ExecuteSize == 0);
    static_assert(HD % ExecuteSize == 0);
    static_assert(KVS % GS == 0);
    static_assert(KVS % Depth == 0);

    const int group_num = (seq_length + QS - 1) / QS;
    const int kv_group_num = num_heads / num_kv_heads;

    sycl::range<3> global_size(bsz, num_heads, group_num * GS);
    sycl::range<3> local_size(1, 1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<KVS * HD * sizeof(fp16) * 2>();
                constexpr int k_slm_offset = 0;
                constexpr int v_slm_offset = KVS * HD * sizeof(fp16);

                const int bsz_idx = item.get_global_id(0);
                const int head_idx = item.get_global_id(1);
                const int kv_head_idx = head_idx / kv_group_num;
                const int32_t seq_idx = item.get_global_id(2);
                const int gid = item.get_group(2);
                const int tid = item.get_local_id(2);

                const int32_t* seq_len = (const int32_t*)seq_lens;
                int32_t seq_bound = seq_len[bsz_idx];

                const int32_t* query_loc = (const int32_t*)query_start_loc;
                int32_t token_idx =
                    query_loc[bsz_idx] + std::min(seq_idx, seq_bound - 1);

                // const IT * query_head = (const IT *)query + token_idx * query_token_stride
                //                                           + head_idx * query_head_stride;
                // const fp16 * key_head = (const fp16 *)key + bsz_idx * key_bsz_stride
                //                                           + kv_head_idx * key_head_stride;
                // const fp16 * value_head = (const fp16 *)value + bsz_idx * value_bsz_stride
                //                                               + kv_head_idx * value_head_stride;
                

                const int* block_tables_ptr = (const int*)block_tables;
                const int* block_table =
                    block_tables_ptr + bsz_idx * block_table_stride_batch;

                const int32_t* context_len_ptr = (const int32_t*)context_lens;
                const int32_t context_len = context_len_ptr[bsz_idx];

                const int32_t token_position = context_len + seq_idx;

                AT * tmp_output_head = (AT *)tmp_output + bsz_idx * tmp_output_bsz_stride
                                                        + head_idx * tmp_output_head_stride;
                IT * output_head = (IT *)output + (query_loc[bsz_idx] + seq_idx) * output_token_stride
                                                + head_idx * output_head_stride;

                simd<int, RepeatCount> seq_idxs;
                #pragma unroll
                for (int i = 0; i < RepeatCount; ++i) {
                    seq_idxs[i] = std::min(gid * QS + tid * RepeatCount + i, seq_length - 1);
                }

                simd<fp16, RepeatCount * HD> query_rows;
                #pragma unroll
                for (int i = 0; i < RepeatCount; ++i) {
                    const IT * query_head = (const IT *)query + seq_idxs[i] * query_token_stride
                                                          + head_idx * query_head_stride;

                    simd<fp16, HD> query_row = block_load<IT, HD>(query_head) * attn_scale;
                    query_rows.template bit_cast_view<fp16, RepeatCount * HD / Depth, Depth>()
                        .template select<HD / Depth, RepeatCount, Depth, 1>(i, 0)
                        = query_row.template bit_cast_view<fp16, HD / Depth, Depth>();
                }

                // const int block_num = is_causal ? (std::min((gid + 1) * QS, seq_length) + KVS - 1) / KVS
                //                                 : (context_length + KVS - 1) / KVS;
                block_num[0] = (std::min((gid + 1) * QS, seq_length) + KVS - 1) / KVS;
                simd<fp16, RepeatCount> max_attns = -sycl::detail::max_v<fp16>();
                simd<fp16, RepeatCount> softmaxs = 0;
                for (int bid = 0; bid < static_cast<int>(block_num[0]); ++bid) {
                    #pragma unroll
                    for (int i = 0; i < RS; ++i) {
                        int which_block = bid * KVS / RS + tid;
                        int which_slot = i;
                        int physical_block_number = block_table[which_block];
                        // simd<fp16, HD> key_row = block_load<fp16, HD>(key_head + (bid * KVS + tid * RS + i) * HD);
                        const fp16 * key_head = (const fp16 *)key +
                            physical_block_number * key_token_stride +
                            kv_head_idx * key_head_stride +
                            which_slot * key_block_stride;
                        simd<fp16, HD> key_row = block_load<fp16, HD>(key_head);
                        slm_block_store<fp16, HD>(k_slm_offset + (tid * RS + i) * HD * sizeof(fp16), key_row);
                    }

                    #pragma unroll
                    for (int i = 0; i < RS; ++i) {
                        int which_block = bid * KVS / RS + tid;
                        int which_slot = i;
                        int physical_block_number = block_table[which_block];
                        // simd<fp16, HD> value_row = block_load<fp16, HD>(value_head + (bid * KVS + tid * RS + i) * HD);
                        const fp16 * value_head = (const fp16 *)value +
                            physical_block_number * value_token_stride +
                            kv_head_idx * value_head_stride +
                            which_slot * value_block_stride;
                        simd<fp16, HD> value_row = block_load<fp16, HD>(value_head);
                        slm_block_store<fp16, HD>(v_slm_offset + (tid * RS + i) * HD * sizeof(fp16), value_row);
                    }

                    barrier();

                    simd<fp16, RepeatCount * KVS> attns;

                    // q @ k
                    #pragma unroll
                    for (int i = 0; i < KVS / ExecuteSize; ++i) {
                        simd<float, RepeatCount * ExecuteSize> sub_attns = 0;
                        #pragma unroll
                        for (int t = 0; t < HD / Depth; ++t) {
                            simd<fp16, RepeatCount * Depth> x =
                                query_rows.template select<RepeatCount * Depth, 1>(t * RepeatCount * Depth);
                            simd<fp16, Depth * ExecuteSize> y =
                                slm_block_load<fp16, Depth * ExecuteSize>(k_slm_offset + (i * HD * ExecuteSize + t * Depth * ExecuteSize) * sizeof(fp16));
                            sub_attns = xmx::dpas<Depth / 2, RepeatCount, float, float, fp16, fp16>(sub_attns, y, x);
                        }

                        if constexpr (Depth == ExecuteSize) {
                            attns.template select<RepeatCount * ExecuteSize, 1>(i * RepeatCount * ExecuteSize) += sub_attns;
                        } else {
                            constexpr int N = Depth / ExecuteSize;
                            attns.template bit_cast_view<fp16, RepeatCount * KVS / ExecuteSize, ExecuteSize>()
                                .template select<RepeatCount, N, ExecuteSize, 1>(i / N * RepeatCount * N + i % N, 0)
                                += sub_attns.template bit_cast_view<float, RepeatCount, ExecuteSize>();
                        }
                    }

                    // softmax
                    simd<fp16, RepeatCount> sub_max_attns;
                    #pragma unroll
                    for (int i = 0; i < RepeatCount; ++i) {
                        sub_max_attns[i] = hmax<fp16, fp16, KVS>(
                            attns.template bit_cast_view<fp16, RepeatCount * KVS / Depth, Depth>()
                                .template select<KVS / Depth, RepeatCount, Depth, 1>(i, 0)
                        );
                    }

                    const simd<fp16, RepeatCount> new_max_attns = max(max_attns, sub_max_attns);
                    const simd<fp16, RepeatCount> scales = exp(max_attns - new_max_attns);
                    max_attns = new_max_attns;

                    #pragma unroll
                    for (int t = 0; t < KVS / Depth; ++t) {
                        #pragma unroll
                        for (int r = 0; r < RepeatCount; ++r) {
                            attns.template select<Depth, 1>(t * RepeatCount * Depth + r * Depth) = exp(
                                attns.template select<Depth, 1>(t * RepeatCount * Depth + r * Depth) - new_max_attns[r]
                            );
                        }
                    }

                    simd<fp16, Depth * ExecuteSize> ones = 1;
                    simd<float, RepeatCount * ExecuteSize> softmax_accs = 0;
                    #pragma unroll
                    for (int t = 0; t < KVS / Depth; ++t) {
                        simd<fp16, RepeatCount * Depth> x =
                            attns.template select<RepeatCount * Depth, 1>(t * RepeatCount * Depth);
                        softmax_accs = xmx::dpas<Depth / 2, RepeatCount, float, float, fp16, fp16>(softmax_accs, ones, x);
                    }
                    simd<fp16, RepeatCount> sub_softmaxs = softmax_accs.template select<RepeatCount, ExecuteSize>(0);
                    softmaxs = softmaxs * scales + sub_softmaxs;

                    // attn @ v
                    const simd<float, RepeatCount> fp32_scales = scales;
#ifdef __linux__
                    #pragma unroll(4)   // reduce register usage on linux
#else
                    #pragma unroll
#endif
                    for (int i = 0; i < HD / ExecuteSize; ++i) {
                        AT * addr = tmp_output_head + (gid * QS + tid * RepeatCount) * HD + i * RepeatCount * ExecuteSize;
                        simd<float, RepeatCount * ExecuteSize> accs = block_load<AT, RepeatCount * ExecuteSize>(addr);
                        #pragma unroll
                        for (int r = 0; r < RepeatCount; ++r) {
                            accs.template select<ExecuteSize, 1>(r * ExecuteSize) *= fp32_scales[r];
                        }
                        #pragma unroll
                        for (int t = 0; t < KVS / Depth; ++t) {
                            simd<fp16, RepeatCount * Depth> x =
                                attns.template select<RepeatCount * Depth, 1>(t * RepeatCount * Depth);
                            simd<fp16, Depth * ExecuteSize> y =
                                slm_block_load<fp16, Depth * ExecuteSize>(v_slm_offset + (i * KVS * ExecuteSize + t * Depth * ExecuteSize) * sizeof(fp16));
                            accs = xmx::dpas<Depth / 2, RepeatCount, float, float, fp16, fp16>(accs, y, x);
                        }
                        block_store<AT, RepeatCount * ExecuteSize>(addr, accs);
                    }

                    barrier();
                }

                #pragma unroll
                for (int i = 0; i < HD / ExecuteSize; ++i) {
                    AT * src_addr = tmp_output_head + (gid * QS + tid * RepeatCount) * HD + i * RepeatCount * ExecuteSize;
                    simd<float, RepeatCount * ExecuteSize> accs = block_load<AT, RepeatCount * ExecuteSize>(src_addr);
                    #pragma unroll
                    for (int r = 0; r < RepeatCount; ++r) {
                        IT * dst_addr = output_head + static_cast<int>(seq_idxs[r]) * HD + i * ExecuteSize;
                        simd<float, ExecuteSize> result = accs.template select<ExecuteSize, 1>(r * ExecuteSize) / static_cast<fp16>(softmaxs[r]);
                        block_store<IT, ExecuteSize>(dst_addr, result);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "chunked prefill xmx kernel");
}

template<const int HD, const int GS, const int KVS,
         const int RepeatCount, const int Depth, const int ExecuteSize>
auto dispatch_chunked_prefill_xmx(ST it) {
    switch (it) {
        case ST::Half: return std::make_tuple(reshape_key_cache_kernel<fp16, Depth, ExecuteSize, HD>,
                                              reshape_value_cache_kernel<fp16, Depth, ExecuteSize, HD, KVS>,
                                              chunked_prefill_xmx_kernel<fp16, bf16, HD, GS, KVS, RepeatCount, Depth, ExecuteSize>);
        default: throw std::runtime_error("unsupported dtype, only fp16 are supported");
    }
}

void chunked_prefill_xmx(
    torch::Tensor output,
    torch::Tensor query,            // [num_tokens, num_heads, head_dim]
    torch::Tensor key,              // [num_tokens, num_kv_heads * head_size]
    torch::Tensor value,            // [num_tokens, num_kv_heads * head_size]
    torch::Tensor block_tables,     // for chunked prefill
    torch::Tensor query_start_loc,  // for chunked prefill
    torch::Tensor context_lens,     // for chunked prefill
    torch::Tensor seq_lens,         // for chunked prefill
    float attn_scale,
    int block_size,                 // for chunked prefill
    int max_input_length,           // for chunked prefill
    int max_context_length,
    int num_kv_heads
) {
    // int64_t bsz = query.size(0);
    // int64_t num_heads = query.size(1);
    // int64_t seq_length = query.size(2);
    // int64_t head_dim = query.size(3);
    // int64_t num_kv_heads = key.size(1);
    // int64_t context_length = key.size(2);

    int64_t num_tokens = query.size(0);
    int64_t num_heads = query.size(1);
    int64_t head_dim = query.size(2);
    int64_t batch_size = seq_lens.size(0);

    constexpr int max_qs = 512;
    constexpr int max_kvs = 128;
    int64_t padding_seq_length = (max_input_length + max_qs - 1) / max_qs * max_qs;
    int64_t padding_context_length = (max_context_length + max_kvs - 1) / max_kvs * max_kvs;

    // TODO(xiangyu): kv cache memory needs to refine

    auto tmp_output = torch::zeros({bsz, num_heads, padding_seq_length, head_dim},
                                   torch::device(query.device()).dtype(query.dtype()));
    // auto output = torch::empty({bsz, num_heads, seq_length, head_dim},
    //                            torch::device(query.device()).dtype(query.dtype()));

    // auto tmp_output = torch::zeros({num_tokens, num_heads, head_dim},
    //                       torch::device(query.device()).dtype(query.dtype()));

    auto output = torch::zeros({num_tokens, num_heads, head_dim},
                          torch::device(query.device()).dtype(query.dtype()));

    auto [k_func, v_func, sdp_func] = [&] () {
        switch (get_gpu_type(query.device())) {
            case GpuType::ARL:
            case GpuType::ARC:
            case GpuType::FLEX: {
                switch (head_dim) {
                    case 64:  return dispatch_chunked_prefill_xmx<64,  64, 64, 8, 16, 8>(query.scalar_type());
                    case 80:  return dispatch_chunked_prefill_xmx<80,  64, 64, 8, 16, 8>(query.scalar_type());
                    case 128: return dispatch_chunked_prefill_xmx<128, 32, 32, 8, 16, 8>(query.scalar_type());
                    default: throw std::runtime_error("unsupported head_dim, only 128, 80 and 64 are supported");
                }
            }
            case GpuType::LNL:
            case GpuType::BMG: {
                switch (head_dim) {
                    case 64:  return dispatch_chunked_prefill_xmx<64,  64, 128, 8, 16, 16>(query.scalar_type());
                    case 80:  return dispatch_chunked_prefill_xmx<80,  64, 128, 8, 16, 16>(query.scalar_type());
                    case 128: return dispatch_chunked_prefill_xmx<128, 32, 64,  8, 16, 16>(query.scalar_type());
                    default: throw std::runtime_error("unsupported head_dim, only 128, 80 and 64 are supported");
                }
            }
            default: throw std::runtime_error("unsupported device, only ARC, FLEX, ARL, LNL and BMG are supported");
        }
    }();

    k_func(
        key.data_ptr(), block_tables.data_ptr(), context_lengths.data_ptr(),
        key.stride(0), key.stride(1), block_tables.stride(0),
        bsz, num_kv_heads, padding_context_length, query.device()
    );

    v_func(
        value.data_ptr(), block_tables.data_ptr(), context_lengths.data_ptr(),
        value.stride(0), value.stride(1), block_tables.stride(0),
        bsz, num_kv_heads, padding_context_length, query.device()
    );

    sdp_func(
        query.data_ptr(), tmp_key.data_ptr(), tmp_value.data_ptr(),
        mask.data_ptr(), tmp_output.data_ptr(), output.data_ptr(),
        query.stride(0), query.stride(1), query.stride(2),
        tmp_key.stride(0), tmp_key.stride(1),
        tmp_value.stride(0), tmp_value.stride(1),
        mask.stride(0), mask.stride(1), mask.stride(2),
        tmp_output.stride(0), tmp_output.stride(1),
        output.stride(0), output.stride(1),
        attn_scale, bsz, num_heads, num_kv_heads, seq_length, context_length,
        query.device()
    );

    return output;
}