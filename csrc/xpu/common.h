#pragma once

#include <sycl.hpp>
#include <torch/extension.h>

typedef union half_t {
    uint16_t u;
    sycl::half f;
} __half_t;

typedef union ufloat32 {
    unsigned u;
    float f;
} __float_t;

#define QK4_0 64
#define QR4_0 2
#define QK4_1 64
#define QR4_1 2
#define QK5_0 64
#define QR5_0 2
#define QK5_1 64
#define QR5_1 2
#define QK8_0 64
#define QR8_0 1
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1)) // 8
#define QKFP8 64
#define QRFP8 1
#define QKFP6 64
// for iq2 quantization
#define WARP_SIZE 32
#define QK_K 256
#define QK4_K 32
#define QR4_K 2
#define QK6_K 16
#define QKFP6_K 16
#define QR2_XXS 8
#define QI2_XXS (QK_K / (4*QR2_XXS)) // 8
#define QR2_XS 8
#define QI2_XS (QK_K / (4*QR2_XS)) // 8
#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K)) // 16
#define QR1_S 8
#define QI1_S (QK_K / (4*QR1_S)) // 8

typedef struct {
    sycl::half d;          // delta
    uint8_t qs[QK4_0 / 2];    // nibbles / quants
} block_q4_0;

typedef struct {
    uint8_t qs[QK4_0 / 2];    // nibbles / quants
} block_q4_0_qs;

typedef struct {
    uint8_t qs[QK4_1 / 2];    // nibbles / quants
} block_q4_1_qs;

typedef struct {
    sycl::half d;              // delta
    sycl::half m;              // min
    uint8_t qs[QK4_1 / 2];     // nibbles / quants
} block_q4_1;

typedef struct {
    sycl::half d;
    uint8_t qh[8];
    uint8_t qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    sycl::half d;          // delta
    sycl::half m;          // min
    uint8_t qh[8];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;

typedef struct {
    sycl::half d;           // delta
    uint8_t qh[8];          // 3-th bit of quants
    uint8_t qs[QK4_0 / 4];  // nibbles / quants
} block_nf3;

typedef struct {
    uint8_t qh[8];          // 3-th bit of quants
    uint8_t qs[QK4_0 / 4];  // nibbles / quants
} block_nf3_qs;

typedef struct {
    float d;       // delta
    int8_t qs[QK8_0];   // quants
} block_q8_0;

typedef struct {
    int8_t qs[QK8_0];   // quants
} block_q8_0_qs;

typedef struct {
    sycl::half d;
    sycl::half sum;
    int8_t  qs[QK8_1];      // quants
} block_q8_1;

typedef struct {
    uint8_t qs[QKFP8];
} block_fp8_qs;

typedef struct {
    float d;
    uint8_t qs[QKFP8];
} block_fp8;

typedef struct {
    sycl::half d;
    uint16_t qs[QK_K/8]; // 32
} block_iq2_xxs;

typedef struct {
    sycl::half d;
    uint16_t qs[QK_K/8]; // 32
    uint8_t  scales[QK_K/32]; // 8
} block_iq2_xs;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    sycl::half d;            // super-block scale for quantized scales
    sycl::half min;          // super-block min for quantized mins
} block_q2_K;

typedef struct {
    sycl::half d;                 // super-block scale for quantized scales
    sycl::half dmin;              // super-block scale for quantized mins
    uint8_t scales[16];           // scales and mins, quantized with 8 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

typedef struct {
    uint8_t qs[QK_K/2];            // 4-bit quants
} block_q4_K_qs;

typedef struct {
    uint8_t qs[QK4_K/2];            // 4-bit quants
} block_q4_K_qs_block;

typedef struct {
    uint8_t scales[16];            // scales and mins, quantized with 8 bits
} block_q4_K_scales;

typedef struct {
    sycl::half d;               // super-block scale for quantized scales
    sycl::half dmin;            // super-block scale for quantized mins
    uint8_t scales[12];         // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    sycl::half d;            // delta
} block_q6_K;

typedef struct {
    uint32_t qh[QK_K/16];      // quants, upper 2 bits
} block_q6_K_qh;

typedef struct {
    uint32_t ql[QK_K/8];      // quants, lower 4 bits
} block_q6_K_ql;

typedef struct {
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
} block_q6_K_scales;

typedef struct {
    uint8_t ql[QK_K/2];       // quants, lower 4 bits
    uint8_t qh[QK_K/4];       // quants, upper 2 bits
    int8_t  scales[QK_K/16];  // scales, quantized with 8 bits
    sycl::half d;            // super-block scale
} block_fp6_K;
static_assert(sizeof(block_fp6_K) == sizeof(sycl::half) + QK_K / 16 + 3*QK_K/4, "wrong fp6_K block size/padding");

typedef struct {
    uint32_t ql[QK_K/8];      // quants, lower 4 bits
} block_fp6_k_ql;

typedef struct {
    uint32_t qh[QK_K/16];     // quants, upper 2 bits
} block_fp6_k_qh;

typedef struct {
    int8_t scales[QK_K/16];  // scales, quantized with 8 bits, 16
} block_fp6_k_scales;

typedef struct {
    uint32_t ql[QKFP6_K/8];     // upper 2 bits, 2
} block_base_fp6_k_ql;

typedef struct {
    uint32_t qh[QKFP6_K/16];     // upper 2 bits, 1
} block_base_fp6_k_qh;

#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

typedef struct {
    sycl::half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

// 1.8125 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 4-bit block scales
} block_iq1_m;

typedef struct {
    uint8_t ql[QKFP6/2];      // lower 4 bits, 32
    uint8_t qh[QKFP6/4];      // upper 2 bits, 16
    sycl::half  d;            // delta
} block_fp6;

typedef struct {
    uint32_t qh[QKFP6/16];     // upper 2 bits, 4
} block_fp6_32_qh;

typedef struct {
    uint32_t ql[QKFP6/8];      // lower 4 bits, 8
} block_fp6_32_ql;

enum ggml_type {
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_NF4 = 10,
    GGML_TYPE_NF3 = 11,
    GGML_TYPE_FP8E4 = 15,
    GGML_TYPE_FP4 = 16,
    GGML_TYPE_FP8E5 = 19,
    GGML_TYPE_IQ2_XXS = 21,
    GGML_TYPE_IQ2_XS = 22,
    GGML_TYPE_Q2_K = 23,
    GGML_TYPE_IQ1_S = 24,
    GGML_TYPE_IQ1_M = 25,
    GGML_TYPE_Q6_K = 26,
    GGML_TYPE_Q4_K = 27,
    GGML_TYPE_Q5_K = 28,
    GGML_TYPE_FP6 = 29,
    GGML_TYPE_FP6_K = 30,
    GGML_TYPE_Q4_0_WOQ = 34,
    GGML_TYPE_COUNT
};

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = QK4_0,
    [GGML_TYPE_Q4_1] = QK4_1,
    [GGML_TYPE_Q5_0] = QK5_0,
    [GGML_TYPE_Q5_1] = QK5_1,
    [GGML_TYPE_NF4]  = QK4_0,
    [GGML_TYPE_NF3]  = QK4_0,
    [GGML_TYPE_Q8_0] = QK8_0,
    [GGML_TYPE_Q8_1] = QK8_1,
    [GGML_TYPE_FP8E4]  = QKFP8,
    [GGML_TYPE_FP4]  = QK4_0,
    [GGML_TYPE_FP6]  = QKFP6,
    [GGML_TYPE_FP8E5]  = QKFP8,
    [GGML_TYPE_IQ2_XXS] = QK_K,
    [GGML_TYPE_IQ2_XS] = QK_K,
    [GGML_TYPE_Q2_K] = QK_K,
    [GGML_TYPE_IQ1_S] = QK_K,
    [GGML_TYPE_IQ1_M] = QK_K,
    [GGML_TYPE_Q6_K] = QK_K,
    [GGML_TYPE_Q4_K] = QK_K,
    [GGML_TYPE_Q5_K] = QK_K,
    [GGML_TYPE_FP6_K] = QK_K,
    [GGML_TYPE_Q4_0_WOQ] = QK4_0,
};

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = sizeof(block_q4_0),
    [GGML_TYPE_Q4_1] = sizeof(block_q4_1),
    [GGML_TYPE_Q5_0] = sizeof(block_q5_1),
    [GGML_TYPE_Q5_1] = sizeof(block_q5_1),
    [GGML_TYPE_NF4]  = sizeof(block_q4_0),
    [GGML_TYPE_NF3]  = sizeof(block_nf3),
    [GGML_TYPE_Q8_0] = sizeof(block_q8_0),
    [GGML_TYPE_Q8_1] = sizeof(block_q8_1),
    [GGML_TYPE_FP8E4]= sizeof(block_fp8),
    [GGML_TYPE_FP4]  = sizeof(block_q4_0),
    [GGML_TYPE_FP6]  = sizeof(block_fp6),
    [GGML_TYPE_FP8E5]  = sizeof(block_fp8),
    [GGML_TYPE_IQ2_XXS] = sizeof(block_iq2_xxs),
    [GGML_TYPE_IQ2_XS] = sizeof(block_iq2_xs),
    [GGML_TYPE_Q2_K] = sizeof(block_q2_K),
    [GGML_TYPE_IQ1_S] = sizeof(block_iq1_s),
    [GGML_TYPE_IQ1_M] = sizeof(block_iq1_m),
    [GGML_TYPE_Q6_K] = sizeof(block_q6_K),
    [GGML_TYPE_Q4_K] = sizeof(block_q4_K),
    [GGML_TYPE_Q5_K] = sizeof(block_q5_K),
    [GGML_TYPE_FP6_K] = sizeof(block_fp6_K),
    [GGML_TYPE_Q4_0_WOQ] = sizeof(block_q4_0),
};
