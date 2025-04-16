#pragma once

#include <torch/extension.h>
#include <ext/intel/esimd.hpp>

using fp16 = sycl::half;

constexpr uint8_t FP16_EXP_OFFSET = 15;
constexpr uint8_t K_EXP_OFFSET = 9;
constexpr uint8_t V_EXP_OFFSET = 12;
constexpr uint8_t K_OFFSET = (FP16_EXP_OFFSET - K_EXP_OFFSET) << 3;
constexpr uint8_t V_OFFSET = (FP16_EXP_OFFSET - V_EXP_OFFSET) << 3;
constexpr uint16_t K_MAX =
    (uint16_t)0x3FC0 + ((uint16_t)(FP16_EXP_OFFSET - K_EXP_OFFSET) << 10);
constexpr uint16_t K_MIN =
    (uint16_t)0x0040 + ((uint16_t)(FP16_EXP_OFFSET - K_EXP_OFFSET) << 10);
constexpr uint16_t V_MAX =
    (uint16_t)0x3FC0 + ((uint16_t)(FP16_EXP_OFFSET - V_EXP_OFFSET) << 10);
constexpr uint16_t V_MIN =
    (uint16_t)0x0040 + ((uint16_t)(FP16_EXP_OFFSET - V_EXP_OFFSET) << 10);

template <const int HD>
ESIMD_INLINE __ESIMD_NS::simd<uint8_t, HD> quantize_key_row(
    __ESIMD_NS::simd<fp16, HD> key_row) {
  const __ESIMD_NS::simd<fp16, HD> kmax = sycl::bit_cast<fp16, uint16_t>(K_MAX);
  const __ESIMD_NS::simd<fp16, HD> kmin = sycl::bit_cast<fp16, uint16_t>(K_MIN);
  __ESIMD_NS::simd<fp16, HD> key =
      __ESIMD_NS::max(__ESIMD_NS::min(__ESIMD_NS::abs(key_row), kmax), kmin);
  key.template bit_cast_view<uint16_t>() <<= 1;
  __ESIMD_NS::simd<uint8_t, HD> sign =
      key_row.template bit_cast_view<uint8_t>().template select<HD, 2>(1) &
      (uint8_t)0x80;
  return (key.template bit_cast_view<uint8_t>().template select<HD, 2>(1) -
          K_OFFSET) |
         sign;
}

template <const int HD>
ESIMD_INLINE __ESIMD_NS::simd<uint8_t, HD> quantize_value_row(
    __ESIMD_NS::simd<fp16, HD> value_row) {
  const __ESIMD_NS::simd<fp16, HD> vmax = sycl::bit_cast<fp16, uint16_t>(V_MAX);
  const __ESIMD_NS::simd<fp16, HD> vmin = sycl::bit_cast<fp16, uint16_t>(V_MIN);
  __ESIMD_NS::simd<fp16, HD> value =
      __ESIMD_NS::max(__ESIMD_NS::min(__ESIMD_NS::abs(value_row), vmax), vmin);
  value.template bit_cast_view<uint16_t>() <<= 1;
  __ESIMD_NS::simd<uint8_t, HD> sign =
      value_row.template bit_cast_view<uint8_t>().template select<HD, 2>(1) &
      (uint8_t)0x80;
  return (value.template bit_cast_view<uint8_t>().template select<HD, 2>(1) -
          V_OFFSET) |
         sign;
}

template <const int HD>
ESIMD_INLINE __ESIMD_NS::simd<fp16, HD> dequantize_key_row(
    const __ESIMD_NS::simd<uint8_t, HD>& key_row) {
  __ESIMD_NS::simd<uint16_t, HD> result = 0x80;
  result.template bit_cast_view<uint8_t>().template select<HD, 2>(1) =
      (key_row & (uint8_t)0x7F) + K_OFFSET;
  result >>= 1;
  __ESIMD_NS::simd<uint8_t, HD> sign = key_row & (uint8_t)0x80;
  result.template bit_cast_view<uint8_t>().template select<HD, 2>(1) |= sign;
  return result.template bit_cast_view<fp16>();
}

template <const int HD>
ESIMD_INLINE __ESIMD_NS::simd<fp16, HD> dequantize_value_row(
    const __ESIMD_NS::simd<uint8_t, HD>& value_row) {
  __ESIMD_NS::simd<uint16_t, HD> result = 0x80;
  result.template bit_cast_view<uint8_t>().template select<HD, 2>(1) =
      (value_row & (uint8_t)0x7F) + V_OFFSET;
  result >>= 1;
  __ESIMD_NS::simd<uint8_t, HD> sign = value_row & (uint8_t)0x80;
  result.template bit_cast_view<uint8_t>().template select<HD, 2>(1) |= sign;
  return result.template bit_cast_view<fp16>();
}