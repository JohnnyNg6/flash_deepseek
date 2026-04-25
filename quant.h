#ifndef QUANT_H
#define QUANT_H
#include <stddef.h>
#include <stdint.h>

#define QK_K 256
#define Q4K_BLOCK_SIZE 144
#define Q6K_BLOCK_SIZE 210

/* Dequantize n elements (n must be multiple of QK_K) from src into dst. */
void dequantize_row_q4_K(const void *src, float *dst, size_t n);
void dequantize_row_q6_K(const void *src, float *dst, size_t n);

/* fp16 -> fp32 helper (IEEE-754 conversion, no hardware dep). */
float fp16_to_fp32(uint16_t h);

#endif
