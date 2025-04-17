#pragma once

#include <sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "common.h"

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr int QK = 64;
constexpr int SBS = 4;

constexpr int BLOCK_SIZES[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0]     = QK / 2,
    [GGML_TYPE_Q4_0_WOQ] = QK / 2,
};

constexpr int SCALE_SIZES[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0]     = sizeof(fp16),
    [GGML_TYPE_Q4_0_WOQ] = sizeof(fp16),
};

template<int QTYPE>
ESIMD_INLINE auto load_qblocks(const uint8_t * weight, const uint8_t * scale);

template<>
ESIMD_INLINE auto load_qblocks<GGML_TYPE_Q4_0>(const uint8_t * weight, const uint8_t * scale) {
    constexpr int BLOCK_SIZE = BLOCK_SIZES[GGML_TYPE_Q4_0];
    simd<uint8_t, BLOCK_SIZE * SBS> ybytes = block_load<uint8_t, BLOCK_SIZE * SBS>(weight);
    const simd<fp16, SBS> scales = block_load<fp16, SBS>((const fp16 *)scale);

    simd<fp16, QK * SBS> yvs;
    #pragma unroll
    for (int i = 0; i < SBS; ++i) {
        simd<uint8_t, QK> uyv;
        uyv.select<QK / 2, 1>(0) = ybytes.template select<QK / 2, 1>(i * QK / 2) & (uint8_t)0xF;
        uyv.select<QK / 2, 1>(QK / 2) = ybytes.template select<QK / 2, 1>(i * QK / 2) >> (uint8_t)4;
        yvs.template select<QK, 1>(i * QK) = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales[i];
    }
    return yvs;
}

template<>
ESIMD_INLINE auto load_qblocks<GGML_TYPE_Q4_0_WOQ>(const uint8_t * weight, const uint8_t * scale) {
    constexpr int BLOCK_SIZE = BLOCK_SIZES[GGML_TYPE_Q4_0_WOQ];
    simd<uint8_t, BLOCK_SIZE * SBS> ybytes = block_load<uint8_t, BLOCK_SIZE * SBS>(weight);
    const simd<fp16, SBS> scales = block_load<fp16, SBS>((const fp16 *)scale);

    simd<fp16, QK * SBS> yvs;
    #pragma unroll
    for (int i = 0; i < SBS; ++i) {
        simd<uint8_t, QK> uyv;
        uyv.select<QK / 2, 2>(0) = ybytes.template select<QK / 2, 1>(i * QK / 2) & (uint8_t)0xF;
        uyv.select<QK / 2, 2>(1) = ybytes.template select<QK / 2, 1>(i * QK / 2) >> (uint8_t)4;
        yvs.template select<QK, 1>(i * QK) = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales[i];
    }
    return yvs;
}

// C++ doesn't support function template partial specialization, so write a new version for SBS=1
template<int QTYPE>
ESIMD_INLINE auto load_qblock(const uint8_t * weight, const uint8_t * scale);

template<>
ESIMD_INLINE auto load_qblock<GGML_TYPE_Q4_0>(const uint8_t * weight, const uint8_t * scale) {
    constexpr int BLOCK_SIZE = BLOCK_SIZES[GGML_TYPE_Q4_0];
    simd<uint8_t, BLOCK_SIZE> ybytes = block_load<uint8_t, BLOCK_SIZE>(weight);
    fp16 scales = *(const fp16 *)scale;

    simd<uint8_t, QK> uyv;
    uyv.select<QK / 2, 1>(0) = ybytes & (uint8_t)0xF;
    uyv.select<QK / 2, 1>(QK / 2) = ybytes >> (uint8_t)4;
    simd<fp16, QK> yv = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales;

    return yv;
}

template<>
ESIMD_INLINE auto load_qblock<GGML_TYPE_Q4_0_WOQ>(const uint8_t * weight, const uint8_t * scale) {
    constexpr int BLOCK_SIZE = BLOCK_SIZES[GGML_TYPE_Q4_0_WOQ];
    simd<uint8_t, BLOCK_SIZE> ybytes = block_load<uint8_t, BLOCK_SIZE>(weight);
    fp16 scales = *(const fp16 *)scale;

    simd<uint8_t, QK> uyv;
    uyv.select<QK / 2, 2>(0) = ybytes & (uint8_t)0xF;
    uyv.select<QK / 2, 2>(1) = ybytes >> (uint8_t)4;
    simd<fp16, QK> yv = (uyv.bit_cast_view<int8_t>() - (int8_t)8) * scales;

    return yv;
}
