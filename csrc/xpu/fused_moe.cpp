#include "utils.h"
#include "base.hpp"

using ST = at::ScalarType;


template <typename IT, const int VS, const int GS, const int ES, const int QTYPE>
static void moe_forward_kernel(
    const void* input_ptr,
    const int64_t* indexs,
    const uint64_t* qweights,
    void * output_ptr,
    const int num_experts,
    const int state_size,
    const int output_size,
    at::Device device
) {
    static_assert(ES == 8 || ES == 16 || ES == 32);
    assert(output_size % VS == 0);

    const int nb = state_size / QK;
    const int nsb = nb / SBS;

    constexpr int BLOCK_SIZE = BLOCK_SIZES[QTYPE];
    constexpr int SCALE_SIZE = SCALE_SIZES[QTYPE];

    sycl::range<2> global_size(num_experts, output_size / VS * GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<GS * VS * sizeof(float)>();

                const int eid = item.get_global_id(0);
                const int tid = item.get_local_id(1);
                const int vid = item.get_group(1) * VS;

                const uint8_t* weight = (const uint8_t *)(qweights[indexs[eid]]);
                // const uint8_t* weight = (const uint8_t *)(qweights);
                const uint8_t* scales = weight + (int64_t)output_size * nb * BLOCK_SIZE;
                const IT* input = static_cast<const IT *>(input_ptr) + eid * state_size;
                IT* output = static_cast<IT *>(output_ptr) + eid * output_size;

                const uint8_t * weight_base = weight + nb * BLOCK_SIZE * vid;
                const uint8_t * scale_base = scales + nb * SCALE_SIZE * vid;

                simd<IT, VS * ES> accvs{};

                for (int s = tid; s < nsb; s += GS) {
                    simd<IT, SBS * QK> xvs = block_load<IT, SBS * QK>(input + s * SBS * QK);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, SBS * QK> yvs = load_qblocks<QTYPE>(
                            weight_base + v * nb * BLOCK_SIZE + s * SBS * BLOCK_SIZE,
                            scale_base + v * nb * SCALE_SIZE + s * SBS * SCALE_SIZE
                        );

                        #pragma unroll
                        for (int i = 0; i < SBS * QK; i += ES) {
                            accvs.template select<ES, 1>(v * ES) +=
                                xvs.template select<ES, 1>(i) *
                                yvs.template select<ES, 1>(i);
                        }
                    }
                }

                for (int b = nsb * SBS + tid; b < nb; b += GS) {
                    simd<IT, QK> xv = block_load<IT, QK>(input + b * QK);

                    #pragma unroll
                    for (int v = 0; v < VS; ++v) {
                        simd<fp16, QK> yv = load_qblock<QTYPE>(
                            weight_base + v * nb * BLOCK_SIZE + b * BLOCK_SIZE,
                            scale_base + v * nb * SCALE_SIZE + b * SCALE_SIZE
                        );

                        #pragma unroll
                        for (int i = 0; i < QK; i += ES) {
                            accvs.template select<ES, 1>(v * ES) +=
                                xv.template select<ES, 1>(i) *
                                yv.template select<ES, 1>(i);
                        }
                    }
                }

                simd<float, VS> accs;
                #pragma unroll
                for(int v = 0; v < VS; ++v) {
                    accs[v] = sycl::ext::intel::esimd::detail::sum<float, IT, ES>(
                        accvs.template select<ES, 1>(v * ES)
                    );
                }

                slm_block_store<float, VS>(tid * VS * sizeof(float), accs);

                barrier();

                if (tid == 0) {
                    #pragma unroll
                    for (int i = 1; i < GS; ++i) {
                        accs += slm_block_load<float, VS>(i * VS * sizeof(float));
                    }

                    block_store<IT, VS>(output + vid, accs);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "moe forward down kernel");
}


template <int QTYPE>
static auto dispatch_moe_forward(ST scalar_t) {
    switch (scalar_t) {
        case ST::Float: return std::make_tuple(moe_forward_kernel<float, 4, 4, 16, QTYPE>);
        case ST::Half: return std::make_tuple(moe_forward_kernel<fp16, 4, 4, 32, QTYPE>);
        default: throw std::runtime_error("unsupported dtype, only fp32 and fp16 are supported");
    }
}


torch::Tensor moe_forward(
    torch::Tensor input,
    torch::Tensor indexs,
    torch::Tensor qweights_attr,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t output_size,
    int64_t qtype
) {
    auto [func] = [&] () {
        switch (qtype) {
            case GGML_TYPE_Q4_0:
                return dispatch_moe_forward<GGML_TYPE_Q4_0>(input.scalar_type());
            case GGML_TYPE_Q4_0_WOQ:
                return dispatch_moe_forward<GGML_TYPE_Q4_0_WOQ>(input.scalar_type());
            default: throw std::runtime_error("unsupported qtype: " + std::to_string(qtype));
        }
    } ();

    int64_t num_experts = indexs.numel();

    torch::Tensor output = torch::zeros({num_experts, output_size},
                                    torch::device(input.device()).dtype(input.dtype()));

    func(
        input.data_ptr(), indexs.data_ptr<int64_t>(),
        qweights_attr.data_ptr<uint64_t>(), output.data_ptr(),
        num_experts, intermediate_size, output_size, input.device()
    );

    return output;
}
