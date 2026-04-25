#include "quant.h"
#include <string.h>

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            // subnormal -> normalize
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void *src, float *y, size_t n) {
    const uint8_t *p = (const uint8_t *)src;
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        uint16_t d_h, dmin_h;
        memcpy(&d_h,    p + 0, 2);
        memcpy(&dmin_h, p + 2, 2);
        const float d    = fp16_to_fp32(d_h);
        const float dmin = fp16_to_fp32(dmin_h);
        const uint8_t *scales = p + 4;
        const uint8_t *qs     = p + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (qs[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (qs[l] >>   4) - m2;
            qs += 32; is += 2;
        }
        p += Q4K_BLOCK_SIZE;
    }
}

void dequantize_row_q6_K(const void *src, float *y, size_t n) {
    const uint8_t *p = (const uint8_t *)src;
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t  *sc = (const int8_t *)(p + 192);
        uint16_t d_h; memcpy(&d_h, p + 208, 2);
        const float d = fp16_to_fp32(d_h);

        for (int n2 = 0; n2 < QK_K; n2 += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        p += Q6K_BLOCK_SIZE;
    }
}
