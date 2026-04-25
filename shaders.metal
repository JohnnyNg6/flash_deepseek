/*
 * shaders.metal — Metal compute shaders for DeepSeek-R1 Q4_K/Q6_K inference
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP16 helper
// ============================================================================
inline float fp16_to_f32_metal(uint16_t h) {
    return float(as_type<half>(h));
}

// ============================================================================
// Q4_K dequant scale/min extraction (matches quant.c)
// ============================================================================
inline void get_scale_min_k4(int j, device const uint8_t *q,
                              thread uint8_t &d, thread uint8_t &m) {
    if (j < 4) {
        d = q[j]     & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

// ============================================================================
// Q4_K matrix-vector multiply
// ============================================================================
// W is stored as raw Q4_K blocks, row-major. Each row has in_dim/256 blocks.
// Block layout: [d:f16][dmin:f16][scales:12B][qs:128B] = 144 bytes per block.
// Dequantized value = d * scale * nibble - dmin * min
//
// Design: 8 SIMD groups of 32 threads = 256 threads per threadgroup.
// Each SIMD group processes one output row.
// Within a group, 32 lanes split Q4_K blocks across the input dimension.
// x is read directly from device memory (L1 cache handles the reuse).

#define ROWS_PER_TG 8

kernel void dequant_matvec_q4k(
    device const uint8_t* W_packed [[buffer(0)]],
    device const float*   x        [[buffer(1)]],
    device float*         out      [[buffer(2)]],
    constant uint&        out_dim  [[buffer(3)]],
    constant uint&        in_dim   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint num_blocks = in_dim / 256;
    uint row_bytes = num_blocks * 144;

    if (row >= out_dim) return;

    device const uint8_t* row_data = W_packed + (uint64_t)row * row_bytes;

    float acc = 0.0f;

    // Each lane processes blocks in a strided pattern
    for (uint blk = simd_lane; blk < num_blocks; blk += 32) {
        device const uint8_t* bp = row_data + blk * 144;

        // Read block header
        uint16_t d_h, dmin_h;
        d_h    = ((uint16_t)bp[1] << 8) | bp[0];
        dmin_h = ((uint16_t)bp[3] << 8) | bp[2];
        float d    = fp16_to_f32_metal(d_h);
        float dmin = fp16_to_f32_metal(dmin_h);

        device const uint8_t* scales_raw = bp + 4;
        device const uint8_t* qs = bp + 16;

        uint x_base = blk * 256;

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is,     scales_raw, sc0, m0);
            get_scale_min_k4(is + 1, scales_raw, sc1, m1);
            float d1 = d * float(sc0);
            float m1v = dmin * float(m0);
            float d2 = d * float(sc1);
            float m2v = dmin * float(m1);

            // Low nibbles: 32 values with d1/m1
            for (int l = 0; l < 32; l++) {
                float val = d1 * float(qs[l] & 0x0F) - m1v;
                acc += val * x[x_base + j + l];
            }
            // High nibbles: 32 values with d2/m2
            for (int l = 0; l < 32; l++) {
                float val = d2 * float(qs[l] >> 4) - m2v;
                acc += val * x[x_base + j + 32 + l];
            }
            qs += 32;
            is += 2;
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Q6_K matrix-vector multiply
// ============================================================================
// Block layout (210 bytes per 256 elements):
//   ql:     128 bytes (lower 4 bits)
//   qh:      64 bytes (upper 2 bits)
//   scales:  16 bytes (16 × int8 sub-block scales)
//   d:        2 bytes (f16 super-block scale)
//
// Dequantized value = d * scale * (6-bit signed value - 32)

kernel void dequant_matvec_q6k(
    device const uint8_t* W_packed [[buffer(0)]],
    device const float*   x        [[buffer(1)]],
    device float*         out      [[buffer(2)]],
    constant uint&        out_dim  [[buffer(3)]],
    constant uint&        in_dim   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint num_blocks = in_dim / 256;
    uint row_bytes = num_blocks * 210;

    if (row >= out_dim) return;

    device const uint8_t* row_data = W_packed + (uint64_t)row * row_bytes;

    float acc = 0.0f;

    for (uint blk = simd_lane; blk < num_blocks; blk += 32) {
        device const uint8_t* bp = row_data + blk * 210;
        device const uint8_t* ql = bp;           // 128 bytes
        device const uint8_t* qh = bp + 128;     //  64 bytes
        device const int8_t*  sc = (device const int8_t*)(bp + 192); // 16 bytes
        uint16_t d_h = ((uint16_t)bp[209] << 8) | bp[208];
        float d = fp16_to_f32_metal(d_h);

        uint x_base = blk * 256;

        // Process 2 halves of 128 elements each
        for (int n2 = 0; n2 < 256; n2 += 128) {
            for (int l = 0; l < 32; l++) {
                int is_idx = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                uint xi = x_base + n2;
                acc += d * float(sc[is_idx + 0]) * float(q1) * x[xi + l +  0];
                acc += d * float(sc[is_idx + 2]) * float(q2) * x[xi + l + 32];
                acc += d * float(sc[is_idx + 4]) * float(q3) * x[xi + l + 64];
                acc += d * float(sc[is_idx + 6]) * float(q4) * x[xi + l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// F32 matrix-vector multiply (for small tensors like exp_probs_b)
// ============================================================================
kernel void matvec_f32(
    device const float* W   [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint& out_dim  [[buffer(3)]],
    constant uint& in_dim   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    device const float* w_row = W + (uint64_t)row * in_dim;
    float acc = 0.0f;
    for (uint i = simd_lane; i < in_dim; i += 32) {
        acc += w_row[i] * x[i];
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) out[row] = sum;
}

// ============================================================================
// SwiGLU activation: out = silu(gate) * up
// ============================================================================
kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = gate[tid];
    out[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

// ============================================================================
// RMS Normalization (two-pass)
// ============================================================================
kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = x[i];
        acc += v * v;
    }
    float sv = simd_sum(acc);
    uint sl = lid % 32, sg = lid / 32;
    if (sl == 0) shared[sg] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v2 = (sl < (tg_size + 31) / 32) ? shared[sl] : 0.0f;
        v2 = simd_sum(v2);
        if (sl == 0) sum_sq[0] = v2;
    }
}

kernel void rms_norm_apply_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}

// ============================================================================
// Residual add: out = a + b
// ============================================================================
kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}

// ============================================================================
// Weighted sum of K expert outputs
// ============================================================================
kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}
