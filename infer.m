


/*
 * infer.m — DeepSeek-R1 Q4_K_M inference on Apple M2 Ultra.
 * Metal routed-expert path + CPU attention / dense / head.
 * Phase C: persistent thread pool + per-expert Metal queue pipeline
 *          + zero-copy expert LRU cache + in-process tokenizer (NEW).
 *
 * Usage:
 *   ./infer --prompt "Explain relativity" --tokens 128 --k 8
 *   ./infer --prompt-tokens prompt.bin --tokens 32 --k 8     # legacy path
 *
 * Env vars:
 *   USE_CPU_EXPERTS=1    disable Metal, use CPU pthread path
 *   MODEL_DIR=...        override default model directory
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <pthread.h>
#include <getopt.h>
#include <errno.h>
#include <glob.h>
#include <stdatomic.h>
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"





/* ===================== Model constants (DeepSeek-R1) ===================== */
#define HIDDEN_DIM          7168
#define NUM_LAYERS          61
#define FIRST_MOE_LAYER     3
#define NUM_ATTN_HEADS      128
#define Q_LORA_RANK         1536
#define KV_LORA_RANK        512
#define QK_NOPE_HEAD_DIM    128
#define QK_ROPE_HEAD_DIM    64
#define QK_HEAD_DIM         192
#define V_HEAD_DIM          128
#define VOCAB_SIZE          129280
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         256
#define MOE_INTERMEDIATE    2048
#define SHARED_INTERMEDIATE 2048
#define DENSE_INTERMEDIATE  18432
#define N_GROUP             8
#define TOPK_GROUP          4
#define ROUTED_SCALE        2.5f
#define ROPE_THETA          10000.0f
#define ROPE_SCALE          40.0f
#define ROPE_ORIG_CTX       4096.0f
#define YARN_MSCALE         1.0f
#define MAX_SEQ             16384

#define Q4K_BLK    144
#define Q6K_BLK    210
#define GATE_BYTES (57344ULL * Q4K_BLK)
#define UP_BYTES   GATE_BYTES
#define DOWN_Q4K_BYTES (57344ULL * Q4K_BLK)
#define DOWN_Q6K_BYTES (57344ULL * Q6K_BLK)
#define EXPERT_SIZE_MAX (GATE_BYTES + UP_BYTES + DOWN_Q6K_BYTES)
#define GATE_OFF   0
#define UP_OFF     GATE_BYTES
#define DOWN_OFF   (GATE_BYTES + UP_BYTES)

#define MAX_K 16

static double now_ms(void){ struct timeval tv; gettimeofday(&tv,NULL);
    return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ========================== Thread pool ================================== */
#define POOL_WORKERS 64
#define POOL_QSIZE   256

typedef struct task_group {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int             remaining;
} task_group;

typedef struct pool_task {
    void (*fn)(void *);
    void *arg;
    task_group *grp;
} pool_task;

static pool_task        g_pool_q[POOL_QSIZE];
static int              g_pool_head = 0, g_pool_tail = 0;
static pthread_mutex_t  g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_pool_cv = PTHREAD_COND_INITIALIZER;
static int              g_pool_shutdown = 0;
static int              g_pool_ready = 0;

static void group_done(task_group *g) {
    pthread_mutex_lock(&g->mu);
    if (--g->remaining == 0) pthread_cond_broadcast(&g->cv);
    pthread_mutex_unlock(&g->mu);
}

static void *pool_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_pool_mu);
        while (g_pool_head == g_pool_tail && !g_pool_shutdown)
            pthread_cond_wait(&g_pool_cv, &g_pool_mu);
        if (g_pool_shutdown && g_pool_head == g_pool_tail) {
            pthread_mutex_unlock(&g_pool_mu);
            return NULL;
        }
        pool_task t = g_pool_q[g_pool_head];
        g_pool_head = (g_pool_head + 1) % POOL_QSIZE;
        pthread_mutex_unlock(&g_pool_mu);

        @autoreleasepool {
            t.fn(t.arg);
        }
        if (t.grp) group_done(t.grp);
    }
}

static void pool_init(void) {
    if (g_pool_ready) return;
    pthread_t th;
    for (int i = 0; i < POOL_WORKERS; i++) {
        if (pthread_create(&th, NULL, pool_worker, NULL) == 0) {
            pthread_detach(th);
        }
    }
    g_pool_ready = 1;
}

static void group_init(task_group *g, int n) {
    pthread_mutex_init(&g->mu, NULL);
    pthread_cond_init(&g->cv, NULL);
    g->remaining = n;
}

static void group_wait_destroy(task_group *g) {
    pthread_mutex_lock(&g->mu);
    while (g->remaining > 0) pthread_cond_wait(&g->cv, &g->mu);
    pthread_mutex_unlock(&g->mu);
    pthread_mutex_destroy(&g->mu);
    pthread_cond_destroy(&g->cv);
}

static void pool_submit(void (*fn)(void*), void *arg, task_group *grp) {
    pthread_mutex_lock(&g_pool_mu);
    int next = (g_pool_tail + 1) % POOL_QSIZE;
    if (next == g_pool_head) {
        pthread_mutex_unlock(&g_pool_mu);
        fn(arg);
        if (grp) group_done(grp);
        return;
    }
    g_pool_q[g_pool_tail].fn  = fn;
    g_pool_q[g_pool_tail].arg = arg;
    g_pool_q[g_pool_tail].grp = grp;
    g_pool_tail = next;
    pthread_cond_signal(&g_pool_cv);
    pthread_mutex_unlock(&g_pool_mu);
}

/* =========================== MoE profiling =============================== */
typedef struct {
    uint64_t t_upload_ns;
    uint64_t t_pread_max_ns;
    uint64_t t_pread_sum_ns;
    uint64_t t_encode_max_ns;
    uint64_t t_encode_sum_ns;
    uint64_t t_pool_wait_ns;
    uint64_t t_gpu_wait_ns;
    uint64_t t_gpu_wait_first_ns;
    uint64_t t_reduce_ns;
    uint64_t bytes_read;
    uint32_t hits;
    uint32_t K;
    uint32_t layer_idx;
} moe_prof_t;

#define MOE_PROF_CAP 4096
static moe_prof_t g_moe_prof[MOE_PROF_CAP];
static _Atomic uint32_t g_moe_prof_head = 0;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void moe_prof_dump_token(int tok_idx, const char *tag) {
    uint32_t head = atomic_load(&g_moe_prof_head);
    const uint32_t N = NUM_LAYERS - FIRST_MOE_LAYER;
    if (head < N) return;

    uint64_t s_upload=0, s_pmax=0, s_psum=0, s_emax=0, s_esum=0;
    uint64_t s_pool=0, s_gpu=0, s_gpu0=0, s_red=0, s_bytes=0;
    uint64_t s_hits = 0, s_total = 0;

    for (uint32_t j = 0; j < N; j++) {
        moe_prof_t *p = &g_moe_prof[(head - N + j) % MOE_PROF_CAP];
        s_upload += p->t_upload_ns;
        s_pmax   += p->t_pread_max_ns;
        s_psum   += p->t_pread_sum_ns;
        s_emax   += p->t_encode_max_ns;
        s_esum   += p->t_encode_sum_ns;
        s_pool   += p->t_pool_wait_ns;
        s_gpu    += p->t_gpu_wait_ns;
        s_gpu0   += p->t_gpu_wait_first_ns;
        s_red    += p->t_reduce_ns;
        s_bytes  += p->bytes_read;
        s_hits   += p->hits;
        s_total  += p->K;
    }
    double ms = 1e-6;
    double eff_bw = (double)s_bytes / ((double)s_psum * 1e-9 + 1e-12) / 1e9;
    uint64_t moe_total = s_upload + s_pool + s_gpu + s_red;

    fprintf(stderr,
        "[prof %s tok=%d] %u MoE layers, cache %llu/%llu (%.1f%%), "
        "%.2f GB read, MoE-only total %.0f ms\n"
        "  upload         %7.1f ms\n"
        "  pool_wait      %7.1f ms  (pread+encode+commit, wall)\n"
        "    pread max    %7.1f ms  (critical path, sum across layers)\n"
        "    pread sum    %7.1f ms  (%.2f GB/s eff over %d workers)\n"
        "    encode max   %7.1f ms\n"
        "    encode sum   %7.1f ms\n"
        "  gpu_wait sum   %7.1f ms\n"
        "    first cb     %7.1f ms\n"
        "  reduce (saxpy) %7.1f ms\n",
        tag, tok_idx, N,
        (unsigned long long)s_hits, (unsigned long long)s_total,
        s_total ? 100.0 * s_hits / s_total : 0.0,
        s_bytes/1e9, moe_total*ms,
        s_upload*ms,
        s_pool*ms,
        s_pmax*ms,
        s_psum*ms, eff_bw, POOL_WORKERS,
        s_emax*ms, s_esum*ms,
        s_gpu*ms, s_gpu0*ms,
        s_red*ms);
}

/* --- coarse per-token timing --- */
static _Atomic uint64_t g_t_mla_ns   = 0;
static _Atomic uint64_t g_t_dense_ns = 0;
static _Atomic uint64_t g_t_moe_ns   = 0;
static _Atomic uint64_t g_t_head_ns  = 0;

/* =========================== Cached layer ptrs =========================== */
typedef struct {
    const gguf_tensor *attn_norm;
    const gguf_tensor *q_a, *q_a_norm, *q_b;
    const gguf_tensor *kv_a, *kv_a_norm, *k_b, *v_b, *kv_b, *o_proj;
    const gguf_tensor *ffn_norm;
    const gguf_tensor *ffn_gate, *ffn_up, *ffn_down;
    const gguf_tensor *gate_inp, *exp_probs_b;
    const gguf_tensor *shared_gate, *shared_up, *shared_down;
} LayerWeights;

static LayerWeights g_lw[NUM_LAYERS];

static const gguf_tensor* find(gguf_manifest *m, const char *fmt, int L){
    char n[256]; snprintf(n,sizeof(n),fmt,L);
    return gguf_find(m,n);
}

static void cache_layer_weights(gguf_manifest *m){
    for (int L=0; L<NUM_LAYERS; L++){
        LayerWeights *w = &g_lw[L];
        w->attn_norm = find(m, "blk.%d.attn_norm.weight", L);
        w->q_a       = find(m, "blk.%d.attn_q_a.weight", L);
        w->q_a_norm  = find(m, "blk.%d.attn_q_a_norm.weight", L);
        w->q_b       = find(m, "blk.%d.attn_q_b.weight", L);
        w->kv_a      = find(m, "blk.%d.attn_kv_a_mqa.weight", L);
        w->kv_a_norm = find(m, "blk.%d.attn_kv_a_norm.weight", L);
        w->k_b       = find(m, "blk.%d.attn_k_b.weight", L);
        w->v_b       = find(m, "blk.%d.attn_v_b.weight", L);
        w->kv_b      = find(m, "blk.%d.attn_kv_b.weight", L);
        w->o_proj    = find(m, "blk.%d.attn_output.weight", L);
        w->ffn_norm  = find(m, "blk.%d.ffn_norm.weight", L);
        if (L < FIRST_MOE_LAYER) {
            w->ffn_gate = find(m, "blk.%d.ffn_gate.weight", L);
            w->ffn_up   = find(m, "blk.%d.ffn_up.weight",   L);
            w->ffn_down = find(m, "blk.%d.ffn_down.weight", L);
        } else {
            w->gate_inp    = find(m, "blk.%d.ffn_gate_inp.weight", L);
            w->exp_probs_b = find(m, "blk.%d.exp_probs_b.bias", L);
            if (!w->exp_probs_b) w->exp_probs_b = find(m, "blk.%d.exp_probs_b.weight", L);
            w->shared_gate = find(m, "blk.%d.ffn_gate_shexp.weight", L);
            w->shared_up   = find(m, "blk.%d.ffn_up_shexp.weight",   L);
            w->shared_down = find(m, "blk.%d.ffn_down_shexp.weight", L);
        }
    }
}




/* ============================ Tensor helpers ============================= */
static float* dequant_tensor_full(const gguf_tensor *t){
    if (!t) return NULL;
    uint64_t n = 1;
    for (uint32_t i=0; i<t->n_dims; i++) n *= t->dims[i];
    float *out = (float*)malloc(n * sizeof(float));
    if (!out) return NULL;
    void *blob = malloc(t->nbytes);
    if (!blob) { free(out); return NULL; }
    if (gguf_tensor_pread(t, blob, t->nbytes, 0) != (int64_t)t->nbytes) {
        fprintf(stderr, "ERR: pread '%s' failed\n", t->name);
        free(blob); free(out); return NULL;
    }
    switch (t->type) {
        case GGML_TYPE_F32: memcpy(out, blob, n*sizeof(float)); break;
        case GGML_TYPE_F16: {
            const uint16_t *p = (const uint16_t*)blob;
            for (uint64_t i=0;i<n;i++) out[i] = fp16_to_fp32(p[i]);
        } break;
        case GGML_TYPE_BF16: {
            const uint16_t *p = (const uint16_t*)blob;
            for (uint64_t i=0;i<n;i++){
                uint32_t b = ((uint32_t)p[i])<<16;
                memcpy(&out[i], &b, 4);
            }
        } break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K(blob, out, n); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K(blob, out, n); break;
        default:
            fprintf(stderr, "ERR: unsupported dtype for '%s': %s\n",
                    t->name, ggml_type_name(t->type));
            free(blob); free(out); return NULL;
    }
    free(blob);
    return out;
}

static inline void mv(const float *A, const float *x, float *y, int m, int n){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f,
                A, n, x, 1, 0.0f, y, 1);
}






/* ============================ Basic ops ================================== */
#include <arm_neon.h>
#include <time.h>

/* Verbose-mode flag, set by --debug. When 0 the entire init/prefill phase
 * has its stdout+stderr redirected to /dev/null in main(); only the
 * streamed assistant text reaches the user's terminal. */
static int g_debug = 0;

static void rms_norm(const float *x, const float *w, float *o, int d){
    float ss=0; for (int i=0;i<d;i++) ss += x[i]*x[i];
    float inv = 1.0f/sqrtf(ss/d + RMS_NORM_EPS);
    if (w) for (int i=0;i<d;i++) o[i] = x[i]*inv*w[i];
    else   for (int i=0;i<d;i++) o[i] = x[i]*inv;
}

static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

static void silu_mul(const float *g, const float *u, float *o, int d){
    for (int i=0;i<d;i++){
        float v = g[i];
        o[i] = (v/(1.0f+expf(-v))) * u[i];
    }
}

static void softmax(float *x, int n){
    float m=x[0]; for (int i=1;i<n;i++) if (x[i]>m) m=x[i];
    float s=0; for (int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    float inv=1.0f/s; for (int i=0;i<n;i++) x[i]*=inv;
}

static int argmax(const float *x, int n){
    int b=0; float v=x[0];
    for (int i=1;i<n;i++) if (x[i]>v){ v=x[i]; b=i; }
    return b;
}

/* ---- Sampling: temperature + top-p --------------------------------------
 * R1 official recommendation: temp=0.6, top_p=0.95. Greedy argmax causes
 * the thinking-mode loop ("Hmm. Hmm. Okay, the user wants...").
 *
 * Implementation: full qsort over VOCAB_SIZE=129280 floats — ~5 ms/token
 * on M2 (negligible vs ~480 ms/token forward). Could be replaced with a
 * top-K min-heap if it ever shows up on profiles. */
typedef struct { float p; int idx; } pi_t;

static int cmp_pi_desc(const void *a, const void *b) {
    float pa = ((const pi_t*)a)->p, pb = ((const pi_t*)b)->p;
    return (pa < pb) - (pa > pb);
}

static uint32_t g_rng = 0xa1b2c3d4u;
static inline float rand_unit(void) {
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return (x & 0xFFFFFFu) * (1.0f / (float)0x1000000);
}

static int sample_top_p(const float *logits, int n, float temp, float top_p) {
    if (temp <= 0.0f) return argmax(logits, n);

    static pi_t *buf = NULL;
    static int   cap = 0;
    if (cap < n) { free(buf); buf = malloc(n * sizeof(*buf)); cap = n; }

    float inv_t = 1.0f / temp;
    float maxv = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = logits[i] * inv_t;
        if (v > maxv) maxv = v;
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float p = expf(logits[i] * inv_t - maxv);
        buf[i].p = p; buf[i].idx = i; sum += p;
    }
    float inv_s = 1.0f / sum;
    for (int i = 0; i < n; i++) buf[i].p *= inv_s;

    qsort(buf, n, sizeof(pi_t), cmp_pi_desc);

    float cum = 0;
    int   cutoff = n;
    for (int i = 0; i < n; i++) {
        cum += buf[i].p;
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    float r = rand_unit() * cum;
    float c = 0;
    for (int i = 0; i < cutoff; i++) {
        c += buf[i].p;
        if (r <= c) return buf[i].idx;
    }
    return buf[cutoff - 1].idx;
}

/* ---- F16 matvec (NEON). A: row-major (m,n) F16; x,y: F32 ---- */
static void sgemv_f16_serial(const __fp16 *A, const float *x, float *y, int m, int n){
    for (int i = 0; i < m; i++){
        const __fp16 *row = A + (size_t)i * n;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 16 <= n; j += 16){
            float16x8_t lo = vld1q_f16(row + j);
            float16x8_t hi = vld1q_f16(row + j + 8);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(lo)),  vld1q_f32(x + j));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(lo)), vld1q_f32(x + j + 4));
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(hi)),  vld1q_f32(x + j + 8));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(hi)), vld1q_f32(x + j + 12));
        }
        for (; j + 8 <= n; j += 8){
            float16x8_t lo = vld1q_f16(row + j);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(lo)),  vld1q_f32(x + j));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(lo)), vld1q_f32(x + j + 4));
        }
        float s = vaddvq_f32(vaddq_f32(a0, a1));
        for (; j < n; j++) s += (float)row[j] * x[j];
        y[i] = s;
    }
}

typedef struct {
    const __fp16 *A;
    const float  *x;
    float        *y;
    int           rs, re, n;
} F16MvJob;

static void f16_mv_worker(void *arg){
    F16MvJob *j = arg;
    sgemv_f16_serial(j->A + (size_t)j->rs * j->n,
                     j->x, j->y + j->rs,
                     j->re - j->rs, j->n);
}

/* NW=16: M2 Ultra has 16 P-cores. mv_f16 calls in MLA / lm_head are pure
 * DRAM bandwidth bound; more parallel readers gets closer to sustained
 * single-process bandwidth. Excess workers no-op for small m. */
static void mv_f16(const __fp16 *A, const float *x, float *y, int m, int n){
    int NW = 16;
    if (m < NW) NW = m;
    F16MvJob jobs[16];
    task_group g; group_init(&g, NW);
    int per = (m + NW - 1) / NW;
    for (int i = 0; i < NW; i++){
        jobs[i].A  = A;
        jobs[i].x  = x;
        jobs[i].y  = y;
        jobs[i].n  = n;
        jobs[i].rs = i * per;
        jobs[i].re = (i + 1) * per;
        if (jobs[i].re > m) jobs[i].re = m;
        pool_submit(f16_mv_worker, &jobs[i], &g);
    }
    group_wait_destroy(&g);
}

/* 16K page-aligned so the same memory can be wrapped as MTLBuffer
 * via newBufferWithBytesNoCopy (zero-copy GPU MLA). */
static __fp16* f32_to_f16_alloc(const float *src, size_t n){
    __fp16 *dst = NULL;
    size_t bytes = n * sizeof(__fp16);
    size_t bytes_aligned = (bytes + 16383) & ~(size_t)16383;
    if (posix_memalign((void**)&dst, 16384, bytes_aligned) != 0) return NULL;
    for (size_t i = 0; i < n; i++) dst[i] = (__fp16)src[i];
    return dst;
}

static __fp16* dequant_tensor_f16(const gguf_tensor *t){
    float *f32 = dequant_tensor_full(t);
    if (!f32) return NULL;
    uint64_t n = 1;
    for (uint32_t i=0; i<t->n_dims; i++) n *= t->dims[i];
    __fp16 *out = f32_to_f16_alloc(f32, n);
    free(f32);
    return out;
}






/* ========================= YaRN RoPE =====================================
 * DeepSeek-V3/R1 uses YaRN with FREQUENCY-DEPENDENT scaling, not uniform
 * 1/scale on every dim.
 *
 * Per DeepseekV3YarnRotaryEmbedding:
 *
 *   freq_extra(i) = base^(-2i/dim)               (no scaling)
 *   freq_inter(i) = base^(-2i/dim) / scale       (full linear scaling)
 *
 *   low  = floor(dim * log(orig_ctx / (β_fast * 2π)) / (2 * log(base)))
 *   high = ceil (dim * log(orig_ctx / (β_slow * 2π)) / (2 * log(base)))
 *
 *   ramp(i) = clamp((i - low) / (high - low), 0, 1)
 *   mask(i) = 1 - ramp(i)
 *   freq(i) = freq_inter * (1 - mask) + freq_extra * mask
 *
 * For DeepSeek-R1 (dim=64, base=10000, orig_ctx=4096, β_fast=32, β_slow=1):
 *   low = 10, high = 23
 *
 * *** RoPE PAIRING: interleaved (NORMAL), NOT split-half (NeoX). ***
 *
 * llama.cpp's `convert_hf_to_gguf.py` does NOT permute q_b_proj or
 * kv_a_proj_with_mqa for DeepseekV2/V3 — their outputs stay in HF's
 * natural interleaved layout: q_pe = [a0, b0, a1, b1, ..., a31, b31].
 * Correspondingly llama.cpp's `llm_build_deepseek2` calls
 *   ggml_rope_ext(..., LLAMA_ROPE_TYPE_NORMAL, ...)
 * which rotates the pair (x[2i], x[2i+1]) with angle pos * freq(i).
 *
 * The previous split-half (NeoX) implementation rotated (x[i], x[i+32]),
 * pairing dim-i's RoPE freq with dim-(i+32)'s — which in interleaved
 * layout actually live at completely different freqs. Net effect: the
 * 64 RoPE channels become positional NOISE. Symptom: model still picks
 * topic-relevant words via q_nope·k_nope but cannot form factual
 * content because positions are unrecoverable. Output degenerates into
 * paraphrase loops of the prompt itself.
 *
 * mscale_outer = yarn_get_mscale(40, mscale_all_dim=1) /
 *                yarn_get_mscale(40, mscale=1) = 1.0
 * → cos/sin are NOT pre-scaled here. The mscale_all_dim^2 = 1.874×
 * factor is applied at the attention softmax_scale (mla_forward). */

static int   g_yarn_low  = -1;
static int   g_yarn_high = -1;

static void yarn_init_once(void) {
    if (g_yarn_low >= 0) return;
    const int   dim   = QK_ROPE_HEAD_DIM;
    const float base  = ROPE_THETA;
    const float orig  = ROPE_ORIG_CTX;
    const float twopi = 2.0f * (float)M_PI;
    float lo_f = ((float)dim * logf(orig / (32.0f * twopi))) / (2.0f * logf(base));
    float hi_f = ((float)dim * logf(orig / ( 1.0f * twopi))) / (2.0f * logf(base));
    int lo = (int)floorf(lo_f);
    int hi = (int)ceilf (hi_f);
    if (lo < 0)             lo = 0;
    if (hi > (dim/2 - 1))   hi = dim/2 - 1;
    if (hi <= lo)           hi = lo + 1;
    g_yarn_low  = lo;
    g_yarn_high = hi;
    fprintf(stderr,
        "[yarn] dim=%d base=%.0f orig=%.0f scale=%.1f → low=%d high=%d "
        "(interleaved pairing)\n",
        dim, base, orig, ROPE_SCALE, lo, hi);
}

static inline float yarn_freq(int i) {
    const int   dim  = QK_ROPE_HEAD_DIM;
    const float base = ROPE_THETA;
    float freq_extra = 1.0f / powf(base, (float)(2*i) / (float)dim);
    float freq_inter = freq_extra / ROPE_SCALE;
    float denom = (float)(g_yarn_high - g_yarn_low);
    if (denom < 1e-3f) denom = 1e-3f;
    float ramp = ((float)i - (float)g_yarn_low) / denom;
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    float mask = 1.0f - ramp;
    return freq_inter * (1.0f - mask) + freq_extra * mask;
}

/* Interleaved (NORMAL) pairing: freq(i) rotates the pair (x[2i], x[2i+1]).
 * Matches ggml's LLAMA_ROPE_TYPE_NORMAL and HF's natural q_b_proj output. */
static void apply_rope_q(float *qrope, int nheads, int pos){
    yarn_init_once();
    const int half = QK_ROPE_HEAD_DIM / 2;
    for (int h = 0; h < nheads; h++){
        float *qh = qrope + h * QK_ROPE_HEAD_DIM;
        for (int i = 0; i < half; i++){
            float freq = yarn_freq(i);
            float ang  = (float)pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float a = qh[2*i], b = qh[2*i + 1];
            qh[2*i]     = a*c - b*s;
            qh[2*i + 1] = a*s + b*c;
        }
    }
}

static void apply_rope_k(float *krope, int pos){
    yarn_init_once();
    const int half = QK_ROPE_HEAD_DIM / 2;
    for (int i = 0; i < half; i++){
        float freq = yarn_freq(i);
        float ang  = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = krope[2*i], b = krope[2*i + 1];
        krope[2*i]     = a*c - b*s;
        krope[2*i + 1] = a*s + b*c;
    }
}










/* =============================== MLA cache =============================== */
typedef struct {
    float *c_kv;
    float *k_rope;
    int len;
} MLACache;

static MLACache* mla_new(void){
    MLACache *c = calloc(1,sizeof(*c));
    c->c_kv   = calloc((size_t)MAX_SEQ*KV_LORA_RANK, 4);
    c->k_rope = calloc((size_t)MAX_SEQ*QK_ROPE_HEAD_DIM, 4);
    return c;
}







/* ============================= MLA attention ============================= */
/* Big F16 matvecs (q_a, q_b, kv_a, o_proj) run on GPU via zero-copy
 * MTLBuffer wrappers around the existing __fp16 allocations.
 * Per-head k_b/v_b stay on CPU (small, well-parallelized already).
 * Stages 2/2b/4 stay on CPU sgemm (operands are small F32 cache). */

/* Forward decls from Metal GPU section. */
static int g_use_gpu;
static void gpu_mv_f16(id<MTLBuffer> A_buf, const float *x, float *y, int m, int n);

typedef struct {
    float  *q_a_norm_w, *kv_a_norm_w, *attn_norm_w;
    __fp16 *q_a, *q_b;
    __fp16 *kv_a;
    __fp16 *k_b;
    __fp16 *v_b;
    __fp16 *o_proj;
    /* GPU zero-copy wrappers (nil if g_use_gpu==0). */
    id<MTLBuffer> q_a_buf;
    id<MTLBuffer> q_b_buf;
    id<MTLBuffer> kv_a_buf;
    id<MTLBuffer> o_proj_buf;
} LayerAttnDense;

static inline void mla_mv_f16(const __fp16 *cpu_w, id<MTLBuffer> gpu_w,
                              const float *x, float *y, int m, int n)
{
    if (g_use_gpu && gpu_w) gpu_mv_f16(gpu_w, x, y, m, n);
    else                    mv_f16(cpu_w, x, y, m, n);
}

#define MLA_NGROUPS 16

typedef struct {
    const LayerAttnDense *w;
    const float          *q_nope;
    float                *Q_abs;
    int                   h_start, h_end;
} StageQabsJob;

static void stage_qabs_worker(void *arg){
    StageQabsJob *j = arg;
    for (int h = j->h_start; h < j->h_end; h++){
        sgemv_f16_serial(
            j->w->k_b + (size_t)h * KV_LORA_RANK * QK_NOPE_HEAD_DIM,
            j->q_nope + (size_t)h * QK_NOPE_HEAD_DIM,
            j->Q_abs  + (size_t)h * KV_LORA_RANK,
            KV_LORA_RANK, QK_NOPE_HEAD_DIM);
    }
}

typedef struct {
    const LayerAttnDense *w;
    const float          *wc_kv;
    float                *attn_out;
    int                   h_start, h_end;
} StageVoutJob;

static void stage_vout_worker(void *arg){
    StageVoutJob *j = arg;
    for (int h = j->h_start; h < j->h_end; h++){
        sgemv_f16_serial(
            j->w->v_b + (size_t)h * V_HEAD_DIM * KV_LORA_RANK,
            j->wc_kv  + (size_t)h * KV_LORA_RANK,
            j->attn_out + (size_t)h * V_HEAD_DIM,
            V_HEAD_DIM, KV_LORA_RANK);
    }
}

typedef struct {
    float *scores;
    int    S;
    int    h_start, h_end;
} StageSoftmaxJob;

static void stage_softmax_worker(void *arg){
    StageSoftmaxJob *j = arg;
    for (int h = j->h_start; h < j->h_end; h++){
        softmax(j->scores + (size_t)h * j->S, j->S);
    }
}

static void mla_forward(LayerAttnDense *w, float *hidden, MLACache *cache, int pos){
    float residual[HIDDEN_DIM];
    memcpy(residual, hidden, sizeof(residual));

    float normed[HIDDEN_DIM];
    rms_norm(hidden, w->attn_norm_w, normed, HIDDEN_DIM);

    float q_a_out[Q_LORA_RANK];
    mla_mv_f16(w->q_a, w->q_a_buf, normed, q_a_out, Q_LORA_RANK, HIDDEN_DIM);
    float q_a_n[Q_LORA_RANK];
    rms_norm(q_a_out, w->q_a_norm_w, q_a_n, Q_LORA_RANK);

    float *q_full = malloc(sizeof(float) * NUM_ATTN_HEADS * QK_HEAD_DIM);
    mla_mv_f16(w->q_b, w->q_b_buf, q_a_n, q_full,
               NUM_ATTN_HEADS * QK_HEAD_DIM, Q_LORA_RANK);

    float *q_nope = malloc(sizeof(float) * NUM_ATTN_HEADS * QK_NOPE_HEAD_DIM);
    float *q_rope = malloc(sizeof(float) * NUM_ATTN_HEADS * QK_ROPE_HEAD_DIM);
    for (int h = 0; h < NUM_ATTN_HEADS; h++){
        memcpy(q_nope + h*QK_NOPE_HEAD_DIM,
               q_full + h*QK_HEAD_DIM, QK_NOPE_HEAD_DIM*sizeof(float));
        memcpy(q_rope + h*QK_ROPE_HEAD_DIM,
               q_full + h*QK_HEAD_DIM + QK_NOPE_HEAD_DIM,
               QK_ROPE_HEAD_DIM*sizeof(float));
    }
    free(q_full);

    float kv_a_out[KV_LORA_RANK + QK_ROPE_HEAD_DIM];
    mla_mv_f16(w->kv_a, w->kv_a_buf, normed, kv_a_out,
               KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_DIM);
    float c_kv_n[KV_LORA_RANK];
    rms_norm(kv_a_out, w->kv_a_norm_w, c_kv_n, KV_LORA_RANK);
    float k_rope[QK_ROPE_HEAD_DIM];
    memcpy(k_rope, kv_a_out + KV_LORA_RANK, QK_ROPE_HEAD_DIM*sizeof(float));

    apply_rope_q(q_rope, NUM_ATTN_HEADS, pos);
    apply_rope_k(k_rope, pos);

    memcpy(cache->c_kv   + (size_t)cache->len*KV_LORA_RANK,     c_kv_n, KV_LORA_RANK*sizeof(float));
    memcpy(cache->k_rope + (size_t)cache->len*QK_ROPE_HEAD_DIM, k_rope, QK_ROPE_HEAD_DIM*sizeof(float));
    cache->len++;

    int S = cache->len;

    /* YaRN-modified softmax scale (DeepSeek-V3/R1).
     * config: rope_scaling.factor=40, mscale_all_dim=1.0
     *   yarn_get_mscale(s, m) = 0.1 * m * log(s) + 1   (for s > 1)
     *   mscale = 0.1 * 1.0 * log(40) + 1 ≈ 1.36909
     *   softmax_scale = (1/sqrt(qk_head_dim)) * mscale^2
     *                 ≈ 0.07217 * 1.87440 ≈ 0.13526
     * Without the mscale^2 (1.874×) factor, attention scores are too small,
     * softmax becomes too flat, and outputs degenerate into looping
     * "Okay, the user wants ... Okay, so I need ..." patterns. */
    float yarn_mscale = 0.1f * logf(ROPE_SCALE) + 1.0f;
    float scale = (yarn_mscale * yarn_mscale) / sqrtf((float)QK_HEAD_DIM);

    const int per_g = (NUM_ATTN_HEADS + MLA_NGROUPS - 1) / MLA_NGROUPS;

    /* Stage 1 (CPU per-head F16 matvec) */
    float *Q_abs = malloc(sizeof(float) * NUM_ATTN_HEADS * KV_LORA_RANK);
    {
        StageQabsJob jobs[MLA_NGROUPS];
        task_group g; group_init(&g, MLA_NGROUPS);
        for (int gi = 0; gi < MLA_NGROUPS; gi++){
            jobs[gi].w = w;
            jobs[gi].q_nope  = q_nope;
            jobs[gi].Q_abs   = Q_abs;
            jobs[gi].h_start = gi * per_g;
            jobs[gi].h_end   = (gi + 1) * per_g;
            if (jobs[gi].h_end > NUM_ATTN_HEADS) jobs[gi].h_end = NUM_ATTN_HEADS;
            pool_submit(stage_qabs_worker, &jobs[gi], &g);
        }
        group_wait_destroy(&g);
    }

    /* Stage 2 + 2b */
    float *scores_T = malloc(sizeof(float) * NUM_ATTN_HEADS * S);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                NUM_ATTN_HEADS, S, KV_LORA_RANK, scale,
                Q_abs,        KV_LORA_RANK,
                cache->c_kv,  KV_LORA_RANK,
                0.0f, scores_T, S);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                NUM_ATTN_HEADS, S, QK_ROPE_HEAD_DIM, scale,
                q_rope,         QK_ROPE_HEAD_DIM,
                cache->k_rope,  QK_ROPE_HEAD_DIM,
                1.0f, scores_T, S);

    /* Stage 3 */
    {
        StageSoftmaxJob jobs[MLA_NGROUPS];
        task_group g; group_init(&g, MLA_NGROUPS);
        for (int gi = 0; gi < MLA_NGROUPS; gi++){
            jobs[gi].scores  = scores_T;
            jobs[gi].S       = S;
            jobs[gi].h_start = gi * per_g;
            jobs[gi].h_end   = (gi + 1) * per_g;
            if (jobs[gi].h_end > NUM_ATTN_HEADS) jobs[gi].h_end = NUM_ATTN_HEADS;
            pool_submit(stage_softmax_worker, &jobs[gi], &g);
        }
        group_wait_destroy(&g);
    }

    /* Stage 4 */
    float *wc_kv_all = malloc(sizeof(float) * NUM_ATTN_HEADS * KV_LORA_RANK);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NUM_ATTN_HEADS, KV_LORA_RANK, S, 1.0f,
                scores_T,    S,
                cache->c_kv, KV_LORA_RANK,
                0.0f, wc_kv_all, KV_LORA_RANK);

    /* Stage 5 (CPU per-head F16 matvec) */
    float *attn_out = malloc(sizeof(float) * NUM_ATTN_HEADS * V_HEAD_DIM);
    {
        StageVoutJob jobs[MLA_NGROUPS];
        task_group g; group_init(&g, MLA_NGROUPS);
        for (int gi = 0; gi < MLA_NGROUPS; gi++){
            jobs[gi].w        = w;
            jobs[gi].wc_kv    = wc_kv_all;
            jobs[gi].attn_out = attn_out;
            jobs[gi].h_start  = gi * per_g;
            jobs[gi].h_end    = (gi + 1) * per_g;
            if (jobs[gi].h_end > NUM_ATTN_HEADS) jobs[gi].h_end = NUM_ATTN_HEADS;
            pool_submit(stage_vout_worker, &jobs[gi], &g);
        }
        group_wait_destroy(&g);
    }

    free(q_nope); free(q_rope);
    free(Q_abs);  free(scores_T); free(wc_kv_all);

    /* Output projection — biggest single matvec, GPU. */
    float o[HIDDEN_DIM];
    mla_mv_f16(w->o_proj, w->o_proj_buf, attn_out, o,
               HIDDEN_DIM, NUM_ATTN_HEADS * V_HEAD_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = residual[i] + o[i];
    free(attn_out);
}




/* ========================== Group-limited routing ======================== */
/* DeepSeek-V3 noaux_tc routing.
 *
 *   scores         = sigmoid(logits)               // unbiased, used for WEIGHTS
 *   scores_choice  = scores + e_score_correction_b // biased,   used for SELECTION
 *
 *   group score = sum of top-2 scores_choice within group
 *   top-4 groups selected; experts in other groups masked to -inf
 *   top-K experts selected by masked scores_choice
 *   weight[k] = scores[ exp_idx[k] ]               // UNBIASED sigmoid
 *   weight   /= sum(weight); weight *= ROUTED_SCALE
 */
static void route(const float *logits, const float *bias, int K,
                  int *exp_idx, float *exp_w)
{
    float scores       [NUM_EXPERTS];   /* sigmoid only — used for weights */
    float scores_choice[NUM_EXPERTS];   /* sigmoid + bias — used for selection */
    for (int i = 0; i < NUM_EXPERTS; i++) {
        scores[i]        = sigmoidf(logits[i]);
        scores_choice[i] = scores[i] + (bias ? bias[i] : 0.0f);
    }

    /* Per-group score = sum of top-2 scores_choice in that group. */
    const int per_group = NUM_EXPERTS / N_GROUP;
    float gscore[N_GROUP];
    for (int g = 0; g < N_GROUP; g++) {
        float t1 = -1e30f, t2 = -1e30f;
        for (int e = 0; e < per_group; e++) {
            float s = scores_choice[g * per_group + e];
            if (s > t1)      { t2 = t1; t1 = s; }
            else if (s > t2) { t2 = s; }
        }
        gscore[g] = t1 + t2;
    }

    /* Top-TOPK_GROUP groups by gscore. */
    int sel_g[TOPK_GROUP];
    for (int k = 0; k < TOPK_GROUP; k++) {
        int   best = -1;
        float bs   = -1e30f;
        for (int g = 0; g < N_GROUP; g++) {
            int seen = 0;
            for (int j = 0; j < k; j++) if (sel_g[j] == g) { seen = 1; break; }
            if (!seen && gscore[g] > bs) { bs = gscore[g]; best = g; }
        }
        sel_g[k] = best;
    }

    /* Mask scores_choice to selected groups (others → -inf so they're never picked). */
    float masked_choice[NUM_EXPERTS];
    for (int i = 0; i < NUM_EXPERTS; i++) masked_choice[i] = -1e30f;
    for (int k = 0; k < TOPK_GROUP; k++) {
        int g = sel_g[k];
        for (int e = 0; e < per_group; e++) {
            int idx = g * per_group + e;
            masked_choice[idx] = scores_choice[idx];
        }
    }

    /* Top-K experts by masked_choice. Track scores_choice as the "key" we
     * compare on; record the index — weights come from `scores` (unbiased)
     * AFTER selection. */
    float topk_key[MAX_K];
    for (int k = 0; k < K; k++) { topk_key[k] = -1e30f; exp_idx[k] = 0; }
    for (int i = 0; i < NUM_EXPERTS; i++) {
        int min_k = 0;
        for (int k = 1; k < K; k++)
            if (topk_key[k] < topk_key[min_k]) min_k = k;
        if (masked_choice[i] > topk_key[min_k]) {
            topk_key[min_k] = masked_choice[i];
            exp_idx[min_k]  = i;
        }
    }

    /* Weights = UNBIASED sigmoid scores at the selected indices,
     * normalized to sum to ROUTED_SCALE. */
    for (int k = 0; k < K; k++) exp_w[k] = scores[exp_idx[k]];
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += exp_w[k];
    if (s > 0.0f) {
        float f = ROUTED_SCALE / s;
        for (int k = 0; k < K; k++) exp_w[k] *= f;
    }
}






/* ==================== Expert streaming (pool pread) ====================== */
typedef struct {
    int fd; void *dst; off_t off; size_t size; ssize_t got;
} PreadJob;

static void pread_job_run(void *arg){
    PreadJob *j = arg;
    size_t got = 0;
    while (got < j->size) {
        ssize_t r = pread(j->fd, (char*)j->dst + got,
                          j->size - got, j->off + got);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
        got += (size_t)r;
    }
    j->got = (ssize_t)got;
}

static void parallel_pread_experts(int layer_fd, size_t expert_size,
                                    const int *exp_idx, int K,
                                    void **bufs)
{
    PreadJob jobs[MAX_K];
    task_group grp; group_init(&grp, K);
    for (int k=0; k<K; k++){
        jobs[k].fd   = layer_fd;
        jobs[k].dst  = bufs[k];
        jobs[k].off  = (off_t)exp_idx[k] * expert_size;
        jobs[k].size = expert_size;
        jobs[k].got  = 0;
        pool_submit(pread_job_run, &jobs[k], &grp);
    }
    group_wait_destroy(&grp);
}

/* ======================= CPU expert compute (fallback) =================== */
typedef struct {
    float *gate_mat;
    float *up_mat;
    float *down_mat;
    float *gate;
    float *up;
    float *act;
    float *eo;
} ExpertScratch;

static ExpertScratch g_scratch[MAX_K];
static int g_scratch_ready = 0;

static void init_expert_scratch(int K) {
    if (g_scratch_ready) return;
    size_t big_w = (size_t)MOE_INTERMEDIATE * HIDDEN_DIM * sizeof(float);
    for (int k = 0; k < K; k++) {
        posix_memalign((void**)&g_scratch[k].gate_mat, 64, big_w);
        posix_memalign((void**)&g_scratch[k].up_mat,   64, big_w);
        posix_memalign((void**)&g_scratch[k].down_mat, 64, big_w);
        posix_memalign((void**)&g_scratch[k].gate, 64, MOE_INTERMEDIATE*sizeof(float));
        posix_memalign((void**)&g_scratch[k].up,   64, MOE_INTERMEDIATE*sizeof(float));
        posix_memalign((void**)&g_scratch[k].act,  64, MOE_INTERMEDIATE*sizeof(float));
        posix_memalign((void**)&g_scratch[k].eo,   64, HIDDEN_DIM*sizeof(float));
    }
    g_scratch_ready = 1;
}

typedef struct {
    const void  *expert_blob;
    int          down_is_q6;
    const float *h_post;
    int          slot;
    float       *out;
} ExpertJob;

static void expert_worker(void *arg) {
    ExpertJob *j = (ExpertJob*)arg;
    ExpertScratch *s = &g_scratch[j->slot];
    const void *g_blob = (const char*)j->expert_blob + GATE_OFF;
    const void *u_blob = (const char*)j->expert_blob + UP_OFF;
    const void *d_blob = (const char*)j->expert_blob + DOWN_OFF;
    dequantize_row_q4_K(g_blob, s->gate_mat, (size_t)MOE_INTERMEDIATE * HIDDEN_DIM);
    dequantize_row_q4_K(u_blob, s->up_mat,   (size_t)MOE_INTERMEDIATE * HIDDEN_DIM);
    if (j->down_is_q6)
        dequantize_row_q6_K(d_blob, s->down_mat, (size_t)HIDDEN_DIM * MOE_INTERMEDIATE);
    else
        dequantize_row_q4_K(d_blob, s->down_mat, (size_t)HIDDEN_DIM * MOE_INTERMEDIATE);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                MOE_INTERMEDIATE, HIDDEN_DIM, 1.0f,
                s->gate_mat, HIDDEN_DIM, j->h_post, 1, 0.0f, s->gate, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                MOE_INTERMEDIATE, HIDDEN_DIM, 1.0f,
                s->up_mat,   HIDDEN_DIM, j->h_post, 1, 0.0f, s->up,   1);
    silu_mul(s->gate, s->up, s->act, MOE_INTERMEDIATE);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                HIDDEN_DIM, MOE_INTERMEDIATE, 1.0f,
                s->down_mat, MOE_INTERMEDIATE, s->act, 1, 0.0f, j->out, 1);
}

typedef struct {
    const float *shared_gate, *shared_up, *shared_down;
    const float *h_post;
    float       *out;
} SharedJob;

static void shared_expert_worker(void *arg) {
    SharedJob *j = (SharedJob*)arg;
    static __thread float sg[SHARED_INTERMEDIATE];
    static __thread float su[SHARED_INTERMEDIATE];
    static __thread float sa[SHARED_INTERMEDIATE];
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                SHARED_INTERMEDIATE, HIDDEN_DIM, 1.0f,
                j->shared_gate, HIDDEN_DIM, j->h_post, 1, 0.0f, sg, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                SHARED_INTERMEDIATE, HIDDEN_DIM, 1.0f,
                j->shared_up,   HIDDEN_DIM, j->h_post, 1, 0.0f, su, 1);
    silu_mul(sg, su, sa, SHARED_INTERMEDIATE);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                HIDDEN_DIM, SHARED_INTERMEDIATE, 1.0f,
                j->shared_down, SHARED_INTERMEDIATE, sa, 1, 0.0f, j->out, 1);
}




/* ============================ Layer state ================================ */
typedef struct {
    LayerAttnDense attn;
    float *ffn_norm_w;
    /* Dense FFN (layers 0-2): F16 weights, parallel NEON matvec.
     * 18432 × 7168 × 2 bytes = 264 MB per matrix × 3 mats × 3 layers = 2.4 GB.
     * Per token reads 9 × 264 MB = 2.4 GB, ~53 ms at ~45 GB/s effective. */
    __fp16 *ffn_gate, *ffn_up, *ffn_down;
    float *gate_inp;
    float *exp_probs_b;
    float *shared_gate, *shared_up, *shared_down;

    int    layer_fd;
    size_t expert_size;
    size_t down_bytes;
    int    down_is_q6;
} Layer;

static Layer g_layer[NUM_LAYERS];
static float  *g_embd_fp32  = NULL;
static float  *g_out_norm_w = NULL;
/* lm_head weights as F16 (was F32). Saves 50% bandwidth on the 3.7 GB matvec.
 * 129280 × 7168 × 2 bytes = 1.85 GB; with NW=16 NEON matvec at ~95 GB/s
 * effective, target time ~20 ms (down from ~80 ms with Accelerate sgemv). */
static __fp16 *g_output_w_f16 = NULL;





/* ======================== Metal GPU MoE ================================== */
static int g_use_gpu = 1;

static id<MTLDevice>                 g_dev = nil;
static id<MTLCommandQueue>           g_queue_slot[MAX_K] = {nil};
static id<MTLLibrary>                g_lib = nil;
static id<MTLComputePipelineState>   g_pso_q4k = nil;
static id<MTLComputePipelineState>   g_pso_q6k = nil;
static id<MTLComputePipelineState>   g_pso_swiglu = nil;

static id<MTLBuffer>                 g_buf_h_post = nil;
static id<MTLBuffer>                 g_buf_gate[MAX_K];
static id<MTLBuffer>                 g_buf_up  [MAX_K];
static id<MTLBuffer>                 g_buf_act [MAX_K];
static id<MTLBuffer>                 g_buf_eo  [MAX_K];
static id<MTLBuffer>                 g_buf_expert_data[MAX_K];

/* MLA GPU path (NEW) */
static id<MTLLibrary>                g_lib_mla = nil;
static id<MTLComputePipelineState>   g_pso_matvec_f16 = nil;
static id<MTLCommandQueue>           g_mla_queue = nil;
static id<MTLBuffer>                 g_buf_mla_x = nil;
static id<MTLBuffer>                 g_buf_mla_y = nil;

static NSMutableArray *g_metal_anchor = nil;

static id<MTLComputePipelineState>
compile_pso_in_lib(id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[metal] missing kernel: %s\n", name.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[metal] pso %s: %s\n",
                name.UTF8String, err.localizedDescription.UTF8String);
        return nil;
    }
    return pso;
}

static id<MTLComputePipelineState>
compile_pso(id<MTLLibrary> lib, NSString *name) {
    return compile_pso_in_lib(lib, name);
}

#define ANCHOR(obj) do {                                                  \
    if ((obj) != nil) {                                                   \
        [g_metal_anchor addObject:(obj)];                                 \
    } else {                                                              \
        fprintf(stderr, "[metal] WARN nil object at %s:%d (" #obj ")\n",  \
                __FILE__, __LINE__);                                      \
    }                                                                     \
} while (0)

/* Inline MLA kernel: F16 weights × F32 vector → F32 vector.
 * One row per threadgroup, 256 threads, threadgroup partial sum reduction. */
static NSString *MLA_KERNEL_SRC = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matvec_f16(\n"
    "    device const half  *A [[buffer(0)]],\n"
    "    device const float *x [[buffer(1)]],\n"
    "    device float       *y [[buffer(2)]],\n"
    "    constant uint      &m [[buffer(3)]],\n"
    "    constant uint      &n [[buffer(4)]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint gid [[threadgroup_position_in_grid]])\n"
    "{\n"
    "    if (gid >= m) return;\n"
    "    threadgroup float partial[256];\n"
    "    float sum = 0.0f;\n"
    "    device const half *row = A + (size_t)gid * n;\n"
    "    for (uint j = tid; j < n; j += 256) {\n"
    "        sum = fma(float(row[j]), x[j], sum);\n"
    "    }\n"
    "    partial[tid] = sum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = 128; s > 0; s >>= 1) {\n"
    "        if (tid < s) partial[tid] += partial[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0) y[gid] = partial[0];\n"
    "}\n";

/* Wrap an existing __fp16 buffer as a zero-copy MTLBuffer.
 * Caller guarantees pointer is 16K page-aligned (f32_to_f16_alloc does this). */
static id<MTLBuffer> wrap_f16_as_buf(const __fp16 *p, size_t n_elements) {
    if (!g_dev || !p) return nil;
    size_t bytes = n_elements * sizeof(__fp16);
    size_t bytes_aligned = (bytes + 16383) & ~(size_t)16383;
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)p
                                               length:bytes_aligned
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!b) {
        fprintf(stderr, "[metal] wrap_f16_as_buf failed (n=%zu, bytes=%zu)\n",
                n_elements, bytes_aligned);
    }
    return b;
}

/* Synchronous F16 matvec on GPU. ~50-150 us per call overhead;
 * called 4× per MLA layer × 61 layers = ~244 calls/token.
 * x and y stay in CPU memory; copied through persistent shared buffers. */
static void gpu_mv_f16(id<MTLBuffer> A_buf, const float *x, float *y, int m, int n) {
    memcpy(g_buf_mla_x.contents, x, (size_t)n * sizeof(float));
    id<MTLCommandBuffer> cb = [g_mla_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pso_matvec_f16];
    [enc setBuffer:A_buf offset:0 atIndex:0];
    [enc setBuffer:g_buf_mla_x offset:0 atIndex:1];
    [enc setBuffer:g_buf_mla_y offset:0 atIndex:2];
    uint32_t mu = (uint32_t)m, nu = (uint32_t)n;
    [enc setBytes:&mu length:sizeof(mu) atIndex:3];
    [enc setBytes:&nu length:sizeof(nu) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(y, g_buf_mla_y.contents, (size_t)m * sizeof(float));
}

static int metal_init(const char *model_path) {
    (void)model_path;
    if (getenv("USE_CPU_EXPERTS")) {
        fprintf(stderr, "[metal] disabled via USE_CPU_EXPERTS env var\n");
        return 0;
    }

    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) {
        fprintf(stderr, "[metal] no Metal device\n");
        return 0;
    }

    g_metal_anchor = [NSMutableArray array];
    ANCHOR(g_dev);

    for (int k = 0; k < MAX_K; k++) {
        g_queue_slot[k] = [g_dev newCommandQueue];
        if (!g_queue_slot[k]) {
            fprintf(stderr, "[metal] newCommandQueue %d failed\n", k);
            return 0;
        }
        ANCHOR(g_queue_slot[k]);
    }

    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal"
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        fprintf(stderr, "[metal] cannot read shaders.metal from CWD: %s\n",
                err ? err.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
    } else {
        opts.fastMathEnabled = YES;
    }
    g_lib = [g_dev newLibraryWithSource:src options:opts error:&err];
    if (!g_lib) {
        fprintf(stderr, "[metal] compile error: %s\n",
                err.localizedDescription.UTF8String);
        return 0;
    }
    ANCHOR(g_lib);

    g_pso_q4k    = compile_pso(g_lib, @"dequant_matvec_q4k");
    g_pso_q6k    = compile_pso(g_lib, @"dequant_matvec_q6k");
    g_pso_swiglu = compile_pso(g_lib, @"swiglu_fused");
    if (!g_pso_q4k || !g_pso_q6k || !g_pso_swiglu) return 0;
    ANCHOR(g_pso_q4k);
    ANCHOR(g_pso_q6k);
    ANCHOR(g_pso_swiglu);

    /* MLA kernel — separate library compiled from inline source. */
    g_lib_mla = [g_dev newLibraryWithSource:MLA_KERNEL_SRC options:opts error:&err];
    if (!g_lib_mla) {
        fprintf(stderr, "[metal] MLA compile error: %s\n",
                err.localizedDescription.UTF8String);
        return 0;
    }
    ANCHOR(g_lib_mla);
    g_pso_matvec_f16 = compile_pso(g_lib_mla, @"matvec_f16");
    if (!g_pso_matvec_f16) return 0;
    ANCHOR(g_pso_matvec_f16);

    g_mla_queue = [g_dev newCommandQueue];
    if (!g_mla_queue) {
        fprintf(stderr, "[metal] mla queue failed\n");
        return 0;
    }
    ANCHOR(g_mla_queue);

    MTLResourceOptions sh = MTLResourceStorageModeShared;
    g_buf_h_post = [g_dev newBufferWithLength:HIDDEN_DIM*4 options:sh];
    ANCHOR(g_buf_h_post);
    for (int k=0; k<MAX_K; k++) {
        g_buf_gate[k] = [g_dev newBufferWithLength:MOE_INTERMEDIATE*4 options:sh];
        g_buf_up[k]   = [g_dev newBufferWithLength:MOE_INTERMEDIATE*4 options:sh];
        g_buf_act[k]  = [g_dev newBufferWithLength:MOE_INTERMEDIATE*4 options:sh];
        g_buf_eo[k]   = [g_dev newBufferWithLength:HIDDEN_DIM*4 options:sh];
        ANCHOR(g_buf_gate[k]);
        ANCHOR(g_buf_up[k]);
        ANCHOR(g_buf_act[k]);
        ANCHOR(g_buf_eo[k]);
    }

    /* MLA persistent input/output staging buffers.
     * Sized for the largest MLA matvec (o_proj: in=16384, out=7168;
     * q_b: in=1536, out=24576). Take the max for each. */
    size_t max_x = 16384;   /* o_proj input */
    size_t max_y = 24576;   /* q_b output  */
    g_buf_mla_x = [g_dev newBufferWithLength:max_x * 4 options:sh];
    g_buf_mla_y = [g_dev newBufferWithLength:max_y * 4 options:sh];
    ANCHOR(g_buf_mla_x);
    ANCHOR(g_buf_mla_y);

    size_t aligned_size = (EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);
    for (int k = 0; k < MAX_K; k++) {
        void *p = NULL;
        if (posix_memalign(&p, 2*1024*1024, aligned_size) != 0 || !p) {
            fprintf(stderr, "[metal] expert_data[%d] posix_memalign failed\n", k);
            return 0;
        }
        memset(p, 0, aligned_size);
        g_buf_expert_data[k] =
            [g_dev newBufferWithBytesNoCopy:p
                                     length:aligned_size
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
        if (!g_buf_expert_data[k]) {
            fprintf(stderr, "[metal] expert_data[%d] newBuffer failed\n", k);
            free(p);
            return 0;
        }
        ANCHOR(g_buf_expert_data[k]);
    }

    fprintf(stderr,
        "[metal] ok: device=%s, %d MoE queues + 1 MLA queue, MLA F16 matvec on GPU\n",
        g_dev.name.UTF8String, MAX_K);
    return 1;
}

/* Walk all layers and wrap the 4 big F16 attention weights as zero-copy
 * MTLBuffers. Must be called AFTER metal_init AND after load_layer_weights. */
static int wrap_attention_weights_for_gpu(void) {
    if (!g_use_gpu) return 0;
    size_t total = 0;
    for (int L = 0; L < NUM_LAYERS; L++) {
        LayerAttnDense *w = &g_layer[L].attn;
        size_t qa_n = (size_t)Q_LORA_RANK * HIDDEN_DIM;
        size_t qb_n = (size_t)NUM_ATTN_HEADS * QK_HEAD_DIM * Q_LORA_RANK;
        size_t kva_n = (size_t)(KV_LORA_RANK + QK_ROPE_HEAD_DIM) * HIDDEN_DIM;
        size_t op_n = (size_t)HIDDEN_DIM * NUM_ATTN_HEADS * V_HEAD_DIM;
        w->q_a_buf    = wrap_f16_as_buf(w->q_a, qa_n);
        w->q_b_buf    = wrap_f16_as_buf(w->q_b, qb_n);
        w->kv_a_buf   = wrap_f16_as_buf(w->kv_a, kva_n);
        w->o_proj_buf = wrap_f16_as_buf(w->o_proj, op_n);
        if (!w->q_a_buf || !w->q_b_buf || !w->kv_a_buf || !w->o_proj_buf) {
            fprintf(stderr, "[metal] layer %d wrap failed\n", L);
            return -1;
        }
        ANCHOR(w->q_a_buf);
        ANCHOR(w->q_b_buf);
        ANCHOR(w->kv_a_buf);
        ANCHOR(w->o_proj_buf);
        total += (qa_n + qb_n + kva_n + op_n) * 2;
    }
    fprintf(stderr,
        "[metal] wrapped %d layers' attention F16 weights as MTLBuffer "
        "(%.1f GB zero-copy)\n",
        NUM_LAYERS, total / 1e9);
    return 0;
}




/* ======================= Expert LRU cache (NEW) ========================== */
/* DELETED. Trust the OS page cache.
 *
 * Why: every custom cache (this LRU, Metal-buffer LRU, malloc cache, LZ4)
 * was slower than letting macOS manage caching via the page cache. The
 * LRU's malloc'd MTLBuffers are anonymous memory → macOS compresses them
 * under pressure → GPU first_cb explodes 180→597 ms.
 *
 * On M2 Ultra 192 GB: F16 weights take ~35 GB, leaving ~150 GB for OS
 * page cache. That holds ~5300 experts (28.5 MB each) — far more than
 * the 3510-slot custom cache, with zero compression overhead.
 *
 * The token-counter stubs below are kept so main()'s reporting code still
 * compiles. They always read 0. Look at moe_prof's gpu_wait timing instead
 * — warm-page reads complete in ~50 ms, cold-page reads in ~700 ms. */

static uint64_t         g_cache_hits_total = 0;
static uint64_t         g_cache_misses_total = 0;
static _Atomic uint64_t g_cache_hits_token = 0;
static _Atomic uint64_t g_cache_misses_token = 0;

static int cache_init(int cap_gb) {
    (void)cap_gb;
    fprintf(stderr,
        "[cache] disabled — trusting OS page cache "
        "(M2 Ultra has ~150 GB free for caching after F16 weights)\n");
    return 0;
}

static void cache_reset_token(void) {
    atomic_store(&g_cache_hits_token, 0);
    atomic_store(&g_cache_misses_token, 0);
}








/* ======================== Per-expert pipeline ============================ */
/* Chunked pread + per-expert pipeline.
 *
 * Each expert's ~28.5 MB pread is split into N_PREAD_CHUNKS sub-preads,
 * dispatched as independent pool tasks. With K=8 experts × N=8 chunks =
 * 64 parallel preads we fully saturate the 64-worker pool during MoE phase.
 *
 * Tuning history:
 *   N=1: pool_wait 315 ms, aggregate ~38 GB/s (1 stream/expert bottleneck).
 *   N=4: pool_wait 190 ms, aggregate ~110 GB/s (55% of DRAM peak).
 *   N=8: target pool_wait ~100 ms, aggregate ~180 GB/s (90% of DRAM peak).
 *
 * Per-thread memcpy on M2 ~3.5 GB/s; aggregate target = N_PREAD_CHUNKS×K×3.5
 * GB/s, capped by DRAM bandwidth (~200 GB/s) and page-cache inode lock.
 *
 * Coordination: each expert has an atomic chunks_left counter starting at
 * N_PREAD_CHUNKS. The chunk that decrements it to zero is "last" and
 * fires that expert's encode+commit. Other experts may still be pread'ing
 * — GPU pipelines naturally on K queues.
 *
 * SSD-cold reads (~5% of bytes) don't benefit from chunking, but don't
 * regress either — they were already serial at SSD bandwidth.
 */

#define N_PREAD_CHUNKS 8

typedef struct ExpertCoord {
    int                                              k;
    int                                              fd;
    char                                            *dst_base;
    off_t                                            off_base;
    size_t                                           total_size;
    _Atomic int                                      chunks_left;
    _Atomic uint64_t                                 t_chunk_max_ns;
    _Atomic uint64_t                                 t_chunk_sum_ns;
    id<MTLBuffer> __unsafe_unretained                buf_expert;
    id<MTLComputePipelineState> __unsafe_unretained  pso_down;
    void                                            *cb_raw;
    uint64_t                                         t_encode_ns;
} ExpertCoord;

typedef struct ChunkJob {
    ExpertCoord *coord;
    size_t       chunk_off;
    size_t       chunk_size;
} ChunkJob;

static inline void enc_matvec_q(id<MTLComputeCommandEncoder> e,
                                id<MTLComputePipelineState> pso,
                                id<MTLBuffer> W, size_t W_off,
                                id<MTLBuffer> x, id<MTLBuffer> out,
                                uint32_t out_dim, uint32_t in_dim)
{
    [e setComputePipelineState:pso];
    [e setBuffer:W offset:W_off atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&out_dim length:sizeof(uint32_t) atIndex:3];
    [e setBytes:&in_dim  length:sizeof(uint32_t) atIndex:4];
    uint32_t n_tg = (out_dim + 7) / 8;
    [e dispatchThreadgroups:MTLSizeMake(n_tg,1,1)
      threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

static inline void enc_swiglu(id<MTLComputeCommandEncoder> e,
                              id<MTLBuffer> gate, id<MTLBuffer> up,
                              id<MTLBuffer> out, uint32_t dim)
{
    [e setComputePipelineState:g_pso_swiglu];
    [e setBuffer:gate offset:0 atIndex:0];
    [e setBuffer:up   offset:0 atIndex:1];
    [e setBuffer:out  offset:0 atIndex:2];
    [e setBytes:&dim length:sizeof(uint32_t) atIndex:3];
    MTLSize tg = MTLSizeMake(256,1,1);
    MTLSize grid = MTLSizeMake(((dim+255)/256)*256,1,1);
    [e dispatchThreads:grid threadsPerThreadgroup:tg];
}

/* Build + commit one expert's CB. Called from whichever pool worker
 * happens to finish that expert's last chunk. Uses queue_slot[k] so
 * concurrent encodes from different workers don't contend. */
static void encode_expert(ExpertCoord *c) {
    int k = c->k;
    uint64_t t0 = now_ns();
    id<MTLCommandBuffer> cb = [g_queue_slot[k] commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc_matvec_q(e, g_pso_q4k, c->buf_expert, GATE_OFF,
                 g_buf_h_post, g_buf_gate[k], MOE_INTERMEDIATE, HIDDEN_DIM);
    enc_matvec_q(e, g_pso_q4k, c->buf_expert, UP_OFF,
                 g_buf_h_post, g_buf_up[k],   MOE_INTERMEDIATE, HIDDEN_DIM);
    enc_swiglu(e, g_buf_gate[k], g_buf_up[k], g_buf_act[k], MOE_INTERMEDIATE);
    enc_matvec_q(e, c->pso_down, c->buf_expert, DOWN_OFF,
                 g_buf_act[k], g_buf_eo[k],   HIDDEN_DIM, MOE_INTERMEDIATE);
    [e endEncoding];
    [cb commit];
    c->t_encode_ns = now_ns() - t0;
    c->cb_raw = (__bridge_retained void *)cb;
}

static void chunk_pread_run(void *argp) {
    ChunkJob *j = argp;
    ExpertCoord *c = j->coord;
    uint64_t t0 = now_ns();
    size_t got = 0;
    while (got < j->chunk_size) {
        ssize_t r = pread(c->fd,
                          c->dst_base + j->chunk_off + got,
                          j->chunk_size - got,
                          c->off_base + (off_t)j->chunk_off + (off_t)got);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
        got += (size_t)r;
    }
    uint64_t dt = now_ns() - t0;

    /* Track per-expert max chunk time (= expert's pread wall, since chunks
     * run in parallel) and sum (for per-thread bandwidth profiling). */
    uint64_t prev_max = atomic_load(&c->t_chunk_max_ns);
    while (dt > prev_max &&
           !atomic_compare_exchange_weak(&c->t_chunk_max_ns, &prev_max, dt)) {}
    atomic_fetch_add(&c->t_chunk_sum_ns, dt);

    /* Last chunk for this expert wins encode+commit duty. */
    int prev = atomic_fetch_sub(&c->chunks_left, 1);
    if (prev == 1) encode_expert(c);
}

static void metal_moe_forward(int L, const float *h_post,
                              const int *exp_idx, const float *exp_w, int K,
                              float *moe_out)
{
    Layer *lyr = &g_layer[L];
    size_t expert_size = lyr->expert_size;
    int down_is_q6 = lyr->down_is_q6;
    id<MTLComputePipelineState> pso_down = down_is_q6 ? g_pso_q6k : g_pso_q4k;

    moe_prof_t P = {0};
    P.layer_idx = L;
    P.K = K;
    P.bytes_read = (uint64_t)K * expert_size;

    uint64_t t_up0 = now_ns();
    memcpy(g_buf_h_post.contents, h_post, HIDDEN_DIM*sizeof(float));
    P.t_upload_ns = now_ns() - t_up0;

    /* ---- Phase 1: K experts × N_PREAD_CHUNKS chunks parallel pread.
     *      Last chunk per expert fires encode + commit inline. ---- */
    ExpertCoord coords[MAX_K];
    ChunkJob    chunks[MAX_K * N_PREAD_CHUNKS];
    size_t chunk_size = (expert_size + N_PREAD_CHUNKS - 1) / N_PREAD_CHUNKS;

    uint64_t t_pool0 = now_ns();
    {
        task_group g; group_init(&g, K * N_PREAD_CHUNKS);
        for (int k = 0; k < K; k++) {
            ExpertCoord *c = &coords[k];
            c->k          = k;
            c->fd         = lyr->layer_fd;
            c->dst_base   = (char*)[g_buf_expert_data[k] contents];
            c->off_base   = (off_t)exp_idx[k] * expert_size;
            c->total_size = expert_size;
            atomic_store(&c->chunks_left,    N_PREAD_CHUNKS);
            atomic_store(&c->t_chunk_max_ns, 0);
            atomic_store(&c->t_chunk_sum_ns, 0);
            c->buf_expert  = g_buf_expert_data[k];
            c->pso_down    = pso_down;
            c->cb_raw      = NULL;
            c->t_encode_ns = 0;
        }
        for (int k = 0; k < K; k++) {
            for (int ci = 0; ci < N_PREAD_CHUNKS; ci++) {
                ChunkJob *j = &chunks[k * N_PREAD_CHUNKS + ci];
                j->coord     = &coords[k];
                j->chunk_off = (size_t)ci * chunk_size;
                size_t sz = chunk_size;
                if (j->chunk_off + sz > expert_size) sz = expert_size - j->chunk_off;
                j->chunk_size = sz;
                pool_submit(chunk_pread_run, j, &g);
            }
        }
        group_wait_destroy(&g);
    }
    P.t_pool_wait_ns = now_ns() - t_pool0;

    /* Profile: pread max = max over experts of (max chunk time within expert)
     *          pread sum = sum over experts of (sum of chunk times) */
    {
        uint64_t pmax = 0, psum = 0, emax = 0, esum = 0;
        for (int k = 0; k < K; k++) {
            uint64_t cm = atomic_load(&coords[k].t_chunk_max_ns);
            uint64_t cs = atomic_load(&coords[k].t_chunk_sum_ns);
            if (cm > pmax) pmax = cm;
            psum += cs;
            if (coords[k].t_encode_ns > emax) emax = coords[k].t_encode_ns;
            esum += coords[k].t_encode_ns;
        }
        P.t_pread_max_ns  = pmax;
        P.t_pread_sum_ns  = psum;
        P.t_encode_max_ns = emax;
        P.t_encode_sum_ns = esum;
    }

    /* ---- Phase 2: wait all K command buffers. ---- */
    for (int k = 0; k < K; k++) {
        uint64_t tw0 = now_ns();
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)coords[k].cb_raw;
        [cb waitUntilCompleted];
        uint64_t tw1 = now_ns();
        P.t_gpu_wait_ns += (tw1 - tw0);
        if (k == 0) P.t_gpu_wait_first_ns = tw1 - tw0;
    }

    /* ---- Phase 3: weighted reduce. ---- */
    uint64_t t_r0 = now_ns();
    memset(moe_out, 0, HIDDEN_DIM*sizeof(float));
    for (int k = 0; k < K; k++) {
        const float *eo = (const float*)g_buf_eo[k].contents;
        cblas_saxpy(HIDDEN_DIM, exp_w[k], eo, 1, moe_out, 1);
    }
    P.t_reduce_ns = now_ns() - t_r0;

    uint32_t pidx = atomic_fetch_add(&g_moe_prof_head, 1) % MOE_PROF_CAP;
    g_moe_prof[pidx] = P;
}





/* =========================== Layer weight load =========================== */
/* Helper: convert per-head k_b from (NOPE, KV_LORA) F32 to (KV_LORA, NOPE) F16. */
static __fp16* k_b_f32_to_kbt_f16(const float *k_f32){
    size_t n = (size_t)NUM_ATTN_HEADS * KV_LORA_RANK * QK_NOPE_HEAD_DIM;
    __fp16 *out = NULL;
    posix_memalign((void**)&out, 64, n * sizeof(__fp16));
    for (int h = 0; h < NUM_ATTN_HEADS; h++){
        const float *src = k_f32 + (size_t)h * QK_NOPE_HEAD_DIM * KV_LORA_RANK;
        __fp16 *dst = out + (size_t)h * KV_LORA_RANK * QK_NOPE_HEAD_DIM;
        for (int i = 0; i < KV_LORA_RANK; i++){
            for (int j = 0; j < QK_NOPE_HEAD_DIM; j++){
                dst[(size_t)i * QK_NOPE_HEAD_DIM + j] =
                    (__fp16)src[(size_t)j * KV_LORA_RANK + i];
            }
        }
    }
    return out;
}

static void load_layer_weights(int L, gguf_manifest *m, const char *model_path){
    (void)m;
    Layer *lyr = &g_layer[L];
    LayerWeights *lw = &g_lw[L];

    lyr->layer_fd    = -1;
    lyr->expert_size = 0;
    lyr->down_bytes  = 0;
    lyr->down_is_q6  = 0;

    /* Norms stay F32 (tiny). */
    lyr->attn.attn_norm_w = dequant_tensor_full(lw->attn_norm);
    lyr->attn.q_a_norm_w  = dequant_tensor_full(lw->q_a_norm);
    lyr->attn.kv_a_norm_w = dequant_tensor_full(lw->kv_a_norm);

    /* Big attention weights → F16. */
    lyr->attn.q_a    = dequant_tensor_f16(lw->q_a);
    lyr->attn.q_b    = dequant_tensor_f16(lw->q_b);
    lyr->attn.kv_a   = dequant_tensor_f16(lw->kv_a);
    lyr->attn.o_proj = dequant_tensor_f16(lw->o_proj);

    /* k_b / v_b: dequant to F32 first, then transpose-k + convert. */
    if (lw->k_b && lw->v_b) {
        float *k_f32 = dequant_tensor_full(lw->k_b);
        float *v_f32 = dequant_tensor_full(lw->v_b);
        if (!k_f32 || !v_f32) { fprintf(stderr,"FATAL layer %d k/v\n",L); abort(); }
        lyr->attn.k_b = k_b_f32_to_kbt_f16(k_f32);
        size_t v_n = (size_t)NUM_ATTN_HEADS * V_HEAD_DIM * KV_LORA_RANK;
        lyr->attn.v_b = f32_to_f16_alloc(v_f32, v_n);
        free(k_f32); free(v_f32);
    } else if (lw->kv_b) {
        float *kv_full = dequant_tensor_full(lw->kv_b);
        if (!kv_full) { fprintf(stderr,"FATAL layer %d kv_b\n",L); abort(); }
        const size_t per_head_src =
            (size_t)(QK_NOPE_HEAD_DIM + V_HEAD_DIM) * KV_LORA_RANK;
        size_t k_n = (size_t)NUM_ATTN_HEADS * QK_NOPE_HEAD_DIM * KV_LORA_RANK;
        size_t v_n = (size_t)NUM_ATTN_HEADS * V_HEAD_DIM       * KV_LORA_RANK;
        float *k_f32 = malloc(k_n * sizeof(float));
        float *v_f32 = malloc(v_n * sizeof(float));
        for (int h = 0; h < NUM_ATTN_HEADS; h++){
            float *src = kv_full + (size_t)h * per_head_src;
            memcpy(k_f32 + (size_t)h * QK_NOPE_HEAD_DIM * KV_LORA_RANK,
                   src,
                   (size_t)QK_NOPE_HEAD_DIM * KV_LORA_RANK * sizeof(float));
            memcpy(v_f32 + (size_t)h * V_HEAD_DIM * KV_LORA_RANK,
                   src + (size_t)QK_NOPE_HEAD_DIM * KV_LORA_RANK,
                   (size_t)V_HEAD_DIM * KV_LORA_RANK * sizeof(float));
        }
        free(kv_full);
        lyr->attn.k_b = k_b_f32_to_kbt_f16(k_f32);
        lyr->attn.v_b = f32_to_f16_alloc(v_f32, v_n);
        free(k_f32); free(v_f32);
        if (L == 0) fprintf(stderr,
            "[init] using combined attn_kv_b, split + F16 at load\n");
    } else {
        fprintf(stderr, "FATAL layer %d: no k_b/v_b/kv_b\n", L);
        abort();
    }

    lyr->ffn_norm_w  = dequant_tensor_full(lw->ffn_norm);

    if (L < FIRST_MOE_LAYER) {
        /* Dense FFN as F16 — saves ~50 ms/token over the 3 dense layers. */
        lyr->ffn_gate = dequant_tensor_f16(lw->ffn_gate);
        lyr->ffn_up   = dequant_tensor_f16(lw->ffn_up);
        lyr->ffn_down = dequant_tensor_f16(lw->ffn_down);
        if (!lyr->ffn_gate || !lyr->ffn_up || !lyr->ffn_down) {
            fprintf(stderr, "FATAL layer %d: ffn dense F16 alloc failed\n", L);
            abort();
        }
    } else {
        lyr->gate_inp    = dequant_tensor_full(lw->gate_inp);
        lyr->exp_probs_b = dequant_tensor_full(lw->exp_probs_b);
        lyr->shared_gate = dequant_tensor_full(lw->shared_gate);
        lyr->shared_up   = dequant_tensor_full(lw->shared_up);
        lyr->shared_down = dequant_tensor_full(lw->shared_down);

        char p[2048];
        snprintf(p,sizeof(p), "%s/packed_experts/layer_%02d.bin", model_path, L);
        lyr->layer_fd = open(p, O_RDONLY);
        if (lyr->layer_fd < 0) {
            fprintf(stderr, "WARN: no %s\n", p);
        } else {
            fcntl(lyr->layer_fd, F_RDAHEAD, 1);
            struct stat st;
            if (fstat(lyr->layer_fd, &st) != 0 || st.st_size % NUM_EXPERTS != 0) {
                fprintf(stderr, "ERR: expert size layer %d\n", L);
                close(lyr->layer_fd); lyr->layer_fd = -1;
            } else {
                lyr->expert_size = (size_t)(st.st_size / NUM_EXPERTS);
                lyr->down_bytes = lyr->expert_size - GATE_BYTES - UP_BYTES;
                if (lyr->down_bytes == DOWN_Q6K_BYTES)      lyr->down_is_q6 = 1;
                else if (lyr->down_bytes == DOWN_Q4K_BYTES) lyr->down_is_q6 = 0;
                else {
                    fprintf(stderr, "ERR: layer %d down_bytes=%zu\n", L, lyr->down_bytes);
                    close(lyr->layer_fd); lyr->layer_fd = -1;
                }
            }
        }
    }
}










/* =============================== Forward pass ============================ */
static void layer_forward(int L, float *hidden, MLACache *mla, int pos,
                           int K, void **expert_bufs)
{
    Layer *lyr = &g_layer[L];

    uint64_t t_mla0 = now_ns();
    mla_forward(&lyr->attn, hidden, mla, pos);
    atomic_fetch_add(&g_t_mla_ns, now_ns() - t_mla0);

    float h_in[HIDDEN_DIM];
    memcpy(h_in, hidden, sizeof(h_in));

    float h_post[HIDDEN_DIM];
    rms_norm(hidden, lyr->ffn_norm_w, h_post, HIDDEN_DIM);

    if (L < FIRST_MOE_LAYER) {
        /* Dense FFN: F16 weights + parallel NEON matvec.
         * Was F32 cblas_sgemv: 528 MB × 3 mats × 3 layers = 4.7 GB / 45 GB/s ≈ 105 ms.
         * Now F16 mv_f16:      264 MB × 3 mats × 3 layers = 2.4 GB / 45 GB/s ≈ 53 ms. */
        uint64_t t_d0 = now_ns();
        static float g_out[DENSE_INTERMEDIATE];
        static float u_out[DENSE_INTERMEDIATE];
        static float a_out[DENSE_INTERMEDIATE];
        mv_f16(lyr->ffn_gate, h_post, g_out, DENSE_INTERMEDIATE, HIDDEN_DIM);
        mv_f16(lyr->ffn_up,   h_post, u_out, DENSE_INTERMEDIATE, HIDDEN_DIM);
        silu_mul(g_out, u_out, a_out, DENSE_INTERMEDIATE);
        float d_out[HIDDEN_DIM];
        mv_f16(lyr->ffn_down, a_out, d_out, HIDDEN_DIM, DENSE_INTERMEDIATE);
        for (int i=0; i<HIDDEN_DIM; i++) hidden[i] = h_in[i] + d_out[i];
        atomic_fetch_add(&g_t_dense_ns, now_ns() - t_d0);
        return;
    }

    static int diag_once = 0;
    if (!diag_once) {
        fprintf(stderr, "[diag] L=%d g_use_gpu=%d layer_fd=%d expert_size=%zu down=%s\n",
            L, g_use_gpu, lyr->layer_fd, lyr->expert_size,
            lyr->down_is_q6 ? "Q6_K" : "Q4_K");
        diag_once = 1;
    }

    uint64_t t_m0 = now_ns();

    float logits[NUM_EXPERTS];
    mv(lyr->gate_inp, h_post, logits, NUM_EXPERTS, HIDDEN_DIM);
    int exp_idx[MAX_K]; float exp_w[MAX_K];
    route(logits, lyr->exp_probs_b, K, exp_idx, exp_w);

    float sd[HIDDEN_DIM];
    SharedJob sj = {
        .shared_gate = lyr->shared_gate,
        .shared_up   = lyr->shared_up,
        .shared_down = lyr->shared_down,
        .h_post      = h_post,
        .out         = sd,
    };
    task_group sh_grp; group_init(&sh_grp, 1);
    pool_submit(shared_expert_worker, &sj, &sh_grp);

    float moe_out[HIDDEN_DIM] = {0};

    if (g_use_gpu && lyr->layer_fd >= 0) {
        metal_moe_forward(L, h_post, exp_idx, exp_w, K, moe_out);
    } else if (lyr->layer_fd >= 0) {
        init_expert_scratch(K);
        parallel_pread_experts(lyr->layer_fd, lyr->expert_size,
                               exp_idx, K, expert_bufs);
        ExpertJob jobs[MAX_K];
        task_group eg; group_init(&eg, K);
        for (int k = 0; k < K; k++) {
            jobs[k].expert_blob = expert_bufs[k];
            jobs[k].down_is_q6  = lyr->down_is_q6;
            jobs[k].h_post      = h_post;
            jobs[k].slot        = k;
            jobs[k].out         = g_scratch[k].eo;
            pool_submit(expert_worker, &jobs[k], &eg);
        }
        group_wait_destroy(&eg);
        for (int k = 0; k < K; k++) {
            cblas_saxpy(HIDDEN_DIM, exp_w[k], g_scratch[k].eo, 1, moe_out, 1);
        }
    }

    group_wait_destroy(&sh_grp);

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = h_in[i] + moe_out[i] + sd[i];
    atomic_fetch_add(&g_t_moe_ns, now_ns() - t_m0);
}







/* ============================= Embedding / head ========================== */
static void embed_lookup(int tok, float *out){
    memcpy(out, g_embd_fp32 + (size_t)tok*HIDDEN_DIM, HIDDEN_DIM*sizeof(float));
}

static void lm_head(const float *h, float *logits){
    /* F16 weights + parallel NEON matvec: 1.85 GB / 95 GB/s ≈ 20 ms,
     * vs ~80 ms with the previous F32 cblas_sgemv. */
    mv_f16(g_output_w_f16, h, logits, VOCAB_SIZE, HIDDEN_DIM);
}



/* ================================= Main ================================== */
static void print_usage(const char *p){
    printf("Usage: %s [--model PATH] [--tokenizer FILE] "
           "(--prompt TEXT | --prompt-tokens FILE) "
           "--tokens N [--k 8] [--cache-gb 64] "
           "[--temp 0.6] [--top-p 0.95] [--debug]\n", p);
}

/* Redirect stdout+stderr to /dev/null. Returns saved fds (caller must
 * call restore_io). On error the saved fds are -1 and IO is unchanged. */
static void silence_io(int *saved_out, int *saved_err) {
    *saved_out = -1; *saved_err = -1;
    fflush(stdout); fflush(stderr);
    int o = dup(STDOUT_FILENO);
    int e = dup(STDERR_FILENO);
    int dn = open("/dev/null", O_WRONLY);
    if (o < 0 || e < 0 || dn < 0) {
        if (o >= 0) close(o);
        if (e >= 0) close(e);
        if (dn >= 0) close(dn);
        return;
    }
    dup2(dn, STDOUT_FILENO);
    dup2(dn, STDERR_FILENO);
    close(dn);
    *saved_out = o; *saved_err = e;
}

static void restore_io(int *saved_out, int *saved_err) {
    fflush(stdout); fflush(stderr);
    if (*saved_out >= 0) {
        dup2(*saved_out, STDOUT_FILENO);
        close(*saved_out); *saved_out = -1;
    }
    if (*saved_err >= 0) {
        dup2(*saved_err, STDERR_FILENO);
        close(*saved_err); *saved_err = -1;
    }
}

/* DeepSeek-R1 chat template patch.
 *
 * The official template ends with `<|Assistant|><think>\n`. Without those
 * three trailing tokens the model never "enters" thinking mode cleanly;
 * it spontaneously emits </think> with no matching <think>, then loops on
 * "Okay, so the user wants...". Symptom: prompt log shows last=201 (the
 * period of "briefly.") instead of 128798 (<think>) or 198 (\n).
 *
 * Token IDs are hard-coded for DeepSeek-R1 (vocab 128k + 1280 specials):
 *   <|User|>      = 128803  (already exposed as tk.user_id)
 *   <|Assistant|> = 128804  (already exposed as tk.assistant_id)
 *   <think>       = 128798
 *   </think>      = 128799
 *   \n            = 198
 *
 * If your GGUF / tokenizer disagrees, adjust the constants below or fix
 * tokenizer_apply_chat_template in tokenizer.c. */
#define R1_TOK_THINK    128798u
#define R1_TOK_END_THK  128799u
#define R1_TOK_NEWLINE  198u

static int patch_r1_prompt_suffix(uint32_t *tokens, int n_prompt, int max_ids,
                                  uint32_t assistant_id) {
    /* Look at the last few tokens; only append what's missing. */
    int has_assist = 0, has_think = 0;
    int lo = n_prompt - 6; if (lo < 0) lo = 0;
    for (int i = lo; i < n_prompt; i++) {
        if (tokens[i] == assistant_id)   has_assist = 1;
        if (tokens[i] == R1_TOK_THINK)   has_think  = 1;
    }
    if (!has_assist) {
        if (n_prompt + 1 > max_ids) return n_prompt;
        tokens[n_prompt++] = assistant_id;
        fprintf(stderr, "[prompt] appended <|Assistant|> (%u)\n", assistant_id);
    }
    if (!has_think) {
        if (n_prompt + 2 > max_ids) return n_prompt;
        tokens[n_prompt++] = R1_TOK_THINK;
        tokens[n_prompt++] = R1_TOK_NEWLINE;
        fprintf(stderr, "[prompt] appended <think>\\n (%u, %u)\n",
                R1_TOK_THINK, R1_TOK_NEWLINE);
    }
    return n_prompt;
}

int main(int argc, char **argv){
@autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *model_path = getenv("MODEL_DIR");
    if (!model_path) model_path =
        "/Users/user11/models/deepseek-r1-q4km/DeepSeek-R1-Q4_K_M";
    const char *prompt_file    = NULL;
    const char *prompt_text    = NULL;
    const char *tokenizer_path = "tokenizer.bin";
    int   max_tokens = 64;
    int   K          = 8;
    int   cache_gb   = 64;
    float temp       = 0.6f;
    float top_p      = 0.95f;

    static struct option opts[] = {
        {"model",         required_argument, 0, 'm'},
        {"prompt-tokens", required_argument, 0, 'p'},
        {"prompt",        required_argument, 0, 'P'},
        {"tokenizer",     required_argument, 0, 'T'},
        {"tokens",        required_argument, 0, 't'},
        {"k",             required_argument, 0, 'k'},
        {"cache-gb",      required_argument, 0, 'c'},
        {"temp",          required_argument, 0, 'X'},
        {"top-p",         required_argument, 0, 'Y'},
        {"debug",         no_argument,       0, 'd'},
        {"help",          no_argument,       0, 'h'},
        {0,0,0,0}
    };
    int c;
    while ((c=getopt_long(argc,argv,"m:p:P:T:t:k:c:X:Y:dh",opts,NULL))!=-1){
        switch(c){
            case 'm': model_path     = optarg;       break;
            case 'p': prompt_file    = optarg;       break;
            case 'P': prompt_text    = optarg;       break;
            case 'T': tokenizer_path = optarg;       break;
            case 't': max_tokens     = atoi(optarg); break;
            case 'k': K              = atoi(optarg); break;
            case 'c': cache_gb       = atoi(optarg); break;
            case 'X': temp           = (float)atof(optarg); break;
            case 'Y': top_p          = (float)atof(optarg); break;
            case 'd': g_debug        = 1;            break;
            case 'h': print_usage(argv[0]); return 0;
        }
    }
    if (!prompt_file && !prompt_text) { print_usage(argv[0]); return 1; }
    if (K > MAX_K) { fprintf(stderr, "K>%d not supported\n", MAX_K); return 1; }

    /* Time-seeded RNG so each run produces fresh sampling. */
    g_rng = (uint32_t)(time(NULL)) ^ ((uint32_t)getpid() * 2654435761u);
    if (g_rng == 0) g_rng = 0xa1b2c3d4u;

    /* ---- Silence init+prefill in non-debug mode. ---- */
    int saved_out = -1, saved_err = -1;
    if (!g_debug) silence_io(&saved_out, &saved_err);

    printf("=== DeepSeek-R1 Q4_K_M (671B / 37B active) ===\n");
    printf("Model:    %s\n", model_path);
    printf("K:        %d routed + 1 shared\n", K);
    printf("Cache:    %d GB (LRU expert cache)\n", cache_gb);
    printf("Sampler:  temp=%.2f top_p=%.2f\n", temp, top_p);

    /* ---- Tokenizer ---- */
    Tokenizer tk;
    int has_tokenizer = (tokenizer_load(&tk, tokenizer_path) == 0);
    if (prompt_text && !has_tokenizer) {
        if (!g_debug) restore_io(&saved_out, &saved_err);
        fprintf(stderr, "ERROR: --prompt requires a tokenizer (tried %s)\n",
                tokenizer_path);
        return 1;
    }

    char pattern[2048];
    snprintf(pattern, sizeof(pattern), "%s/DeepSeek-R1-Q4_K_M-*-of-*.gguf", model_path);
    glob_t gl;
    if (glob(pattern, GLOB_NOSORT, NULL, &gl) != 0 || gl.gl_pathc == 0) {
        if (!g_debug) restore_io(&saved_out, &saved_err);
        fprintf(stderr, "No shards found: %s\n", pattern); return 1;
    }
    int n_shards = (int)gl.gl_pathc;
    const char **paths = malloc(n_shards * sizeof(*paths));
    for (int i=0;i<n_shards;i++) paths[i] = gl.gl_pathv[i];
    for (int i=0;i<n_shards-1;i++) for (int j=i+1;j<n_shards;j++)
        if (strcmp(paths[i], paths[j]) > 0) {
            const char *t=paths[i]; paths[i]=paths[j]; paths[j]=t;
        }

    gguf_manifest mf;
    double t0 = now_ms();
    if (gguf_load_manifest(&mf, paths, n_shards) != 0) {
        if (!g_debug) restore_io(&saved_out, &saved_err);
        fprintf(stderr, "manifest load failed\n");
        return 1;
    }
    printf("[init] manifest: %d shards, %llu tensors (%.0f ms)\n",
           n_shards, (unsigned long long)mf.total_tensors, now_ms()-t0);

    cache_layer_weights(&mf);

    printf("[init] loading embedding + output norm/head ...\n");
    const gguf_tensor *t_embd = gguf_find(&mf, "token_embd.weight");
    const gguf_tensor *t_out_norm = gguf_find(&mf, "output_norm.weight");
    const gguf_tensor *t_out = gguf_find(&mf, "output.weight");
    g_embd_fp32   = dequant_tensor_full(t_embd);
    g_out_norm_w  = dequant_tensor_full(t_out_norm);
    g_output_w_f16 = dequant_tensor_f16(t_out);

    printf("[init] loading layer weights (dequantized; ~35 GB) ...\n");
    double tL = now_ms();
    for (int i=0;i<NUM_LAYERS;i++){
        load_layer_weights(i, &mf, model_path);
        if ((i+1)%10==0 || i==NUM_LAYERS-1)
            fprintf(stderr, "  layer %2d / %d (%.1fs)\n",
                    i+1, NUM_LAYERS, (now_ms()-tL)/1000.0);
    }

    g_use_gpu = metal_init(model_path);
    if (!g_use_gpu) {
        fprintf(stderr, "[init] *** Metal unavailable — using CPU expert path ***\n");
    } else {
        fprintf(stderr, "[init] *** Metal routed experts enabled ***\n");
        if (cache_init(cache_gb) != 0) {
            if (!g_debug) restore_io(&saved_out, &saved_err);
            fprintf(stderr, "[init] cache init failed\n");
            return 1;
        }
        if (wrap_attention_weights_for_gpu() != 0) {
            if (!g_debug) restore_io(&saved_out, &saved_err);
            fprintf(stderr, "[init] GPU MLA wrap failed\n");
            return 1;
        }
    }

    pool_init();
    fprintf(stderr, "[init] thread pool: %d workers, queue=%d\n",
            POOL_WORKERS, POOL_QSIZE);

    /* ---- Build prompt token stream ---- */
    uint32_t *tokens = NULL;
    int n_prompt = 0;
    const int max_ids = 16384;
    if (prompt_text) {
        tokens = malloc((size_t)max_ids * sizeof(uint32_t));
        n_prompt = tokenizer_apply_chat_template(&tk, prompt_text, tokens, max_ids);
        if (n_prompt <= 0) {
            if (!g_debug) restore_io(&saved_out, &saved_err);
            fprintf(stderr, "tokenization failed\n"); return 1;
        }
        printf("[prompt] tokenized %d tokens from --prompt (raw template)\n", n_prompt);

        /* *** R1 chat-template repair *** — see comment above
         * patch_r1_prompt_suffix. Required for the model to enter
         * thinking-then-answer mode instead of looping on "Okay, so..." */
        n_prompt = patch_r1_prompt_suffix(tokens, n_prompt, max_ids, tk.assistant_id);
    } else {
        FILE *pf = fopen(prompt_file, "rb");
        if (!pf){
            if (!g_debug) restore_io(&saved_out, &saved_err);
            perror("prompt-tokens"); return 1;
        }
        uint32_t n_prompt_u = 0;
        fread(&n_prompt_u, 4, 1, pf);
        n_prompt = (int)n_prompt_u;
        if (n_prompt <= 0 || n_prompt > 131072) {
            if (!g_debug) restore_io(&saved_out, &saved_err);
            fprintf(stderr, "bad n_prompt\n"); return 1;
        }
        tokens = malloc((size_t)n_prompt * sizeof(uint32_t));
        fread(tokens, 4, n_prompt, pf);
        fclose(pf);
    }
    printf("[prompt] %d tokens, first=%u last=%u\n",
           n_prompt, tokens[0], tokens[n_prompt-1]);

    MLACache **mlas = calloc(NUM_LAYERS, sizeof(*mlas));
    for (int i=0;i<NUM_LAYERS;i++) mlas[i] = mla_new();

    void *expert_bufs[16];
    for (int k=0;k<K;k++) posix_memalign(&expert_bufs[k], 2*1024*1024, EXPERT_SIZE_MAX);

    float *hidden = malloc(HIDDEN_DIM*sizeof(float));
    float *logits = malloc(VOCAB_SIZE*sizeof(float));

    printf("\n[prefill]\n");
    int pos = 0;
    double t_pre = now_ms();
    cache_reset_token();
    for (int i=0; i<n_prompt; i++){
        embed_lookup(tokens[i], hidden);
        for (int L=0; L<NUM_LAYERS; L++){
            layer_forward(L, hidden, mlas[L], pos, K, expert_bufs);
        }
        pos++;
        if ((i+1)%8==0 || i==n_prompt-1) {
            fprintf(stderr, "  tok %d/%d (%.1fs)\n",
                    i+1, n_prompt, (now_ms()-t_pre)/1000.0);
        }
    }
    double prefill_s = (now_ms()-t_pre)/1000.0;
    printf("[prefill] %.1fs, %.2f tok/s\n", prefill_s, n_prompt/prefill_s);
    moe_prof_dump_token(n_prompt-1, "prefill");

    /* First generated token (sampled, not argmax). */
    float normed[HIDDEN_DIM];
    rms_norm(hidden, g_out_norm_w, normed, HIDDEN_DIM);
    lm_head(normed, logits);
    int next_tok = sample_top_p(logits, VOCAB_SIZE, temp, top_p);

    /* ---- Restore IO so streamed text reaches the terminal. ---- */
    if (!g_debug) restore_io(&saved_out, &saved_err);

    if (g_debug) {
        fprintf(stderr, "\n[gen] streaming output below "
                        "(stderr lines mixed in for diagnostics)\n");
        fprintf(stderr, "----------------------------------------------------------\n");
    }

    /* Print first generated token. */
    if (has_tokenizer && (uint32_t)next_tok != tk.eos_id) {
        char buf[256];
        int wlen = tokenizer_decode_one(&tk, (uint32_t)next_tok, buf, sizeof(buf));
        if (wlen > 0) fwrite(buf, 1, wlen, stdout);
    } else if (!has_tokenizer) {
        printf("<%d>", next_tok);
    }
    fflush(stdout);

    double t_gen = now_ms();
    int ngen = 0;
    for (int g=0; g<max_tokens; g++){
        atomic_store(&g_t_mla_ns,   0);
        atomic_store(&g_t_dense_ns, 0);
        atomic_store(&g_t_moe_ns,   0);
        atomic_store(&g_t_head_ns,  0);
        cache_reset_token();

        double ts = now_ms();
        embed_lookup(next_tok, hidden);
        for (int L=0; L<NUM_LAYERS; L++){
            layer_forward(L, hidden, mlas[L], pos, K, expert_bufs);
        }
        pos++;

        uint64_t t_h0 = now_ns();
        rms_norm(hidden, g_out_norm_w, normed, HIDDEN_DIM);
        lm_head(normed, logits);
        next_tok = sample_top_p(logits, VOCAB_SIZE, temp, top_p);
        atomic_fetch_add(&g_t_head_ns, now_ns() - t_h0);

        int is_eos = 0;
        if (has_tokenizer) {
            if ((uint32_t)next_tok == tk.eos_id) is_eos = 1;
        } else {
            if (next_tok == 0 || next_tok == 1) is_eos = 1;
        }

        if (!is_eos) {
            if (has_tokenizer) {
                char buf[256];
                int wlen = tokenizer_decode_one(&tk, (uint32_t)next_tok,
                                                buf, sizeof(buf));
                if (wlen > 0) fwrite(buf, 1, wlen, stdout);
            } else {
                printf("<%d>", next_tok);
            }
            fflush(stdout);
        }

        if (g_debug) {
            double tok_ms   = now_ms() - ts;
            double mla_ms   = atomic_load(&g_t_mla_ns)  /1e6;
            double dense_ms = atomic_load(&g_t_dense_ns)/1e6;
            double moe_ms   = atomic_load(&g_t_moe_ns)  /1e6;
            double head_ms  = atomic_load(&g_t_head_ns) /1e6;
            fprintf(stderr,
                "\n[%3d] token=%d  %.0f ms (%.3f tok/s)  "
                "mla=%.0f dense=%.0f moe=%.0f head=%.0f\n",
                g, next_tok, tok_ms, 1000.0/tok_ms,
                mla_ms, dense_ms, moe_ms, head_ms);
            if (g % 4 == 3 || g == max_tokens - 1) {
                moe_prof_dump_token(g, "gen");
            }
        }

        ngen++;
        if (is_eos) {
            if (g_debug) fprintf(stderr, "[gen] EOS\n");
            break;
        }
    }

    /* Trailing newline so the shell prompt isn't on the same line as text. */
    fputc('\n', stdout);
    fflush(stdout);

    if (g_debug) {
        double gen_s = (now_ms()-t_gen)/1000.0;
        fprintf(stderr, "----------------------------------------------------------\n");
        fprintf(stderr, "[gen] %d tokens in %.1fs (%.3f tok/s)\n",
                ngen, gen_s, ngen/gen_s);
    }

    if (has_tokenizer) tokenizer_free(&tk);
    for (int i=0;i<NUM_LAYERS;i++){
        free(mlas[i]->c_kv); free(mlas[i]->k_rope); free(mlas[i]);
    }
    free(mlas);
    free(tokens);
    gguf_free_manifest(&mf);
    free(paths); globfree(&gl);
    return 0;
}
}







