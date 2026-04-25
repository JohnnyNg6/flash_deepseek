/*
 * gguf_dump.c — Dump a multi-shard GGUF manifest for verification.
 *
 * Usage:
 *   ./gguf_dump <shard1.gguf> <shard2.gguf> ...
 *
 * Prints:
 *   1. Per-shard summary (version, alignment, n_kv, n_tensors, data_start).
 *   2. Key architecture metadata (from shard 0).
 *   3. Sorted tensor table: name, shard, shape, type, abs_offset, nbytes.
 *   4. Aggregate stats by tensor-name prefix.
 *   5. Sanity checks for DeepSeek-R1 expected tensors.
 */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static int cmp_tensor_name(const void *a, const void *b) {
    const gguf_tensor *ta = *(const gguf_tensor *const *)a;
    const gguf_tensor *tb = *(const gguf_tensor *const *)b;
    return strcmp(ta->name, tb->name);
}

static void print_kv_scalar(const gguf_kv *kv) {
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   printf("%" PRIu8,  kv->v.u8);  break;
        case GGUF_TYPE_INT8:    printf("%" PRId8,  kv->v.i8);  break;
        case GGUF_TYPE_UINT16:  printf("%" PRIu16, kv->v.u16); break;
        case GGUF_TYPE_INT16:   printf("%" PRId16, kv->v.i16); break;
        case GGUF_TYPE_UINT32:  printf("%" PRIu32, kv->v.u32); break;
        case GGUF_TYPE_INT32:   printf("%" PRId32, kv->v.i32); break;
        case GGUF_TYPE_UINT64:  printf("%" PRIu64, kv->v.u64); break;
        case GGUF_TYPE_INT64:   printf("%" PRId64, kv->v.i64); break;
        case GGUF_TYPE_FLOAT32: printf("%g",       kv->v.f32); break;
        case GGUF_TYPE_FLOAT64: printf("%g",       kv->v.f64); break;
        case GGUF_TYPE_BOOL:    printf("%s", kv->v.b ? "true" : "false"); break;
        case GGUF_TYPE_STRING:  printf("\"%.80s%s\"",
                                      kv->v.str,
                                      strlen(kv->v.str) > 80 ? "..." : ""); break;
        default: printf("(?)"); break;
    }
}

static void print_kv(const gguf_kv *kv) {
    printf("  %-50s %-7s ", kv->key, gguf_value_type_name(kv->type));
    if (kv->type == GGUF_TYPE_ARRAY) {
        printf("[%s x %" PRIu64 "]",
               gguf_value_type_name(kv->arr.elem_type),
               kv->arr.n);
        if (kv->arr.elem_type == GGUF_TYPE_STRING && kv->arr.n > 0) {
            char **ss = kv->arr.data;
            printf(" e.g. \"%.40s%s\"",
                   ss[0], strlen(ss[0]) > 40 ? "..." : "");
        }
    } else {
        print_kv_scalar(kv);
    }
    printf("\n");
}

/* Check that a required tensor exists and has expected type/shape */
static int check_tensor(const gguf_manifest *m, const char *name,
                        ggml_type want_type, int expect_dims) {
    const gguf_tensor *t = gguf_find(m, name);
    if (!t) { printf("  MISSING: %s\n", name); return 0; }
    int ok = 1;
    if (want_type != GGML_TYPE_COUNT && t->type != want_type) {
        printf("  TYPE MISMATCH: %s is %s, expected %s\n",
               name, ggml_type_name(t->type), ggml_type_name(want_type));
        ok = 0;
    }
    if (expect_dims > 0 && (int)t->n_dims != expect_dims) {
        printf("  DIMS MISMATCH: %s has %u dims, expected %d\n",
               name, t->n_dims, expect_dims);
        ok = 0;
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <shard1.gguf> [shard2.gguf ...]\n", argv[0]);
        return 1;
    }

    const char **paths = (const char**)&argv[1];
    int n_paths = argc - 1;

    gguf_manifest m;
    if (gguf_load_manifest(&m, paths, n_paths) < 0) {
        fprintf(stderr, "failed to load manifest\n");
        return 1;
    }

    /* --- Per-shard summary --- */
    printf("=== Shards ===\n");
    uint64_t grand_bytes = 0;
    for (int i = 0; i < m.n_shards; i++) {
        const gguf_shard *s = &m.shards[i];
        printf("  [%2d] %s\n", i, s->path);
        printf("       version=%u  align=%" PRIu64 "  n_kv=%" PRIu64
               "  n_tensors=%" PRIu64 "  data_start=%" PRIu64
               "  file_size=%" PRIu64 "\n",
               s->version, s->alignment, s->n_kv, s->n_tensors,
               s->data_start, s->file_size);
        grand_bytes += s->file_size;
    }
    printf("  total file bytes = %" PRIu64 " (%.2f GB)\n\n",
           grand_bytes, grand_bytes / 1e9);

    /* --- Architecture metadata (from shard 0) --- */
    printf("=== Key metadata (shard 0) ===\n");
    static const char *KEYS[] = {
        "general.architecture",
        "general.name",
        "general.alignment",
        "general.file_type",
        "general.quantization_version",
        "deepseek2.block_count",
        "deepseek2.context_length",
        "deepseek2.embedding_length",
        "deepseek2.feed_forward_length",
        "deepseek2.attention.head_count",
        "deepseek2.attention.head_count_kv",
        "deepseek2.attention.key_length",
        "deepseek2.attention.value_length",
        "deepseek2.attention.q_lora_rank",
        "deepseek2.attention.kv_lora_rank",
        "deepseek2.rope.dimension_count",
        "deepseek2.rope.freq_base",
        "deepseek2.rope.scaling.type",
        "deepseek2.rope.scaling.factor",
        "deepseek2.rope.scaling.original_context_length",
        "deepseek2.rope.scaling.yarn_log_multiplier",
        "deepseek2.expert_count",
        "deepseek2.expert_used_count",
        "deepseek2.expert_shared_count",
        "deepseek2.expert_feed_forward_length",
        "deepseek2.expert_shared_feed_forward_length",
        "deepseek2.expert_weights_scale",
        "deepseek2.expert_weights_norm",
        "deepseek2.expert_gating_func",
        "deepseek2.leading_dense_block_count",
        "deepseek2.vocab_size",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
    };
    for (size_t i = 0; i < sizeof(KEYS)/sizeof(KEYS[0]); i++) {
        const gguf_kv *kv = gguf_find_kv(&m.shards[0], KEYS[i]);
        if (kv) print_kv(kv);
    }
    printf("\n");

    /* --- Full tensor list (sorted) --- */
    gguf_tensor **sorted = malloc(m.total_tensors * sizeof(gguf_tensor*));
    memcpy(sorted, m.all_tensors, m.total_tensors * sizeof(gguf_tensor*));
    qsort(sorted, m.total_tensors, sizeof(gguf_tensor*), cmp_tensor_name);

    printf("=== Tensors (%" PRIu64 " total) ===\n", m.total_tensors);
    printf("  %-60s %5s  %-6s  %-28s  %14s  %14s\n",
           "name", "shard", "type", "shape", "abs_offset", "nbytes");
    uint64_t tensor_bytes = 0;
    for (uint64_t i = 0; i < m.total_tensors; i++) {
        const gguf_tensor *t = sorted[i];
        char shape[32];
        int n = snprintf(shape, sizeof(shape), "[");
        for (uint32_t d = 0; d < t->n_dims; d++) {
            n += snprintf(shape + n, sizeof(shape) - n, "%s%" PRIu64,
                          d ? "," : "", t->dims[d]);
        }
        snprintf(shape + n, sizeof(shape) - n, "]");
        printf("  %-60s %5d  %-6s  %-28s  %14" PRIu64 "  %14" PRIu64 "\n",
               t->name, t->shard_idx, ggml_type_name(t->type), shape,
               t->abs_offset, t->nbytes);
        tensor_bytes += t->nbytes;
    }
    printf("  tensor bytes = %" PRIu64 " (%.2f GB)\n\n",
           tensor_bytes, tensor_bytes / 1e9);

    /* --- Aggregate by prefix --- */
    /* A simple bucketing: group by tensor-name class. */
    typedef struct { const char *pat; uint64_t bytes; uint64_t count; } bucket;
    bucket B[] = {
        { "token_embd",    0, 0 },
        { "output_norm",   0, 0 },
        { "output",        0, 0 },
        { "attn_q_a",      0, 0 },
        { "attn_q_b",      0, 0 },
        { "attn_kv_a",     0, 0 },
        { "attn_kv_b",     0, 0 },
        { "attn_k_b",      0, 0 },
        { "attn_v_b",      0, 0 },
        { "attn_output",   0, 0 },
        { "attn_norm",     0, 0 },
        { "attn_q_a_norm", 0, 0 },
        { "attn_kv_a_norm",0, 0 },
        { "ffn_gate_inp",  0, 0 },
        { "ffn_gate_exps", 0, 0 },
        { "ffn_up_exps",   0, 0 },
        { "ffn_down_exps", 0, 0 },
        { "ffn_gate_shexp",0, 0 },
        { "ffn_up_shexp",  0, 0 },
        { "ffn_down_shexp",0, 0 },
        { "ffn_gate",      0, 0 },
        { "ffn_up",        0, 0 },
        { "ffn_down",      0, 0 },
        { "ffn_norm",      0, 0 },
        { "exp_probs_b",   0, 0 },
    };
    int NB = sizeof(B)/sizeof(B[0]);
    uint64_t other_bytes = 0, other_count = 0;
    for (uint64_t i = 0; i < m.total_tensors; i++) {
        const gguf_tensor *t = m.all_tensors[i];
        int matched = 0;
        for (int b = 0; b < NB; b++) {
            if (strstr(t->name, B[b].pat)) {
                B[b].bytes += t->nbytes;
                B[b].count += 1;
                matched = 1;
                break;
            }
        }
        if (!matched) { other_bytes += t->nbytes; other_count++; }
    }
    printf("=== Tensor classes ===\n");
    printf("  %-20s %8s %14s %10s\n", "pattern", "count", "bytes", "GB");
    for (int b = 0; b < NB; b++) {
        if (B[b].count)
            printf("  %-20s %8" PRIu64 " %14" PRIu64 " %9.2f\n",
                   B[b].pat, B[b].count, B[b].bytes, B[b].bytes / 1e9);
    }
    if (other_count)
        printf("  %-20s %8" PRIu64 " %14" PRIu64 " %9.2f\n",
               "(other)", other_count, other_bytes, other_bytes / 1e9);
    printf("\n");

    /* --- DeepSeek-R1 sanity checks --- */
    printf("=== DeepSeek-R1 structural checks ===\n");
    int ok = 1;
    ok &= check_tensor(&m, "token_embd.weight",  GGML_TYPE_COUNT, 2);
    ok &= check_tensor(&m, "output_norm.weight", GGML_TYPE_F32,   1);
    ok &= check_tensor(&m, "output.weight",      GGML_TYPE_COUNT, 2);

    /* Expected: 61 blocks, layers 0..2 dense, 3..60 MoE */
    const gguf_kv *kv_n = gguf_find_kv(&m.shards[0], "deepseek2.block_count");
    int n_blocks = kv_n ? (int)kv_n->v.u32 : 61;
    const gguf_kv *kv_dense = gguf_find_kv(&m.shards[0], "deepseek2.leading_dense_block_count");
    int n_dense = kv_dense ? (int)kv_dense->v.u32 : 3;

    printf("  checking %d blocks (first %d dense, rest MoE)\n", n_blocks, n_dense);
    char nm[128];
    for (int L = 0; L < n_blocks; L++) {
        /* Per-layer attention + norms (MLA) */
        static const char *mla_attn[] = {
            "attn_norm.weight",
            "attn_q_a.weight",        "attn_q_a_norm.weight",
            "attn_q_b.weight",
            "attn_kv_a_mqa.weight",   "attn_kv_a_norm.weight",
            "attn_kv_b.weight",
            "attn_output.weight",
        };
        for (size_t k = 0; k < sizeof(mla_attn)/sizeof(mla_attn[0]); k++) {
            snprintf(nm, sizeof(nm), "blk.%d.%s", L, mla_attn[k]);
            if (!gguf_find(&m, nm)) { printf("  MISSING: %s\n", nm); ok = 0; }
        }
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", L);
        if (!gguf_find(&m, nm)) { printf("  MISSING: %s\n", nm); ok = 0; }

        if (L < n_dense) {
            /* Dense FFN */
            static const char *dense[] = {
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            };
            for (size_t k = 0; k < sizeof(dense)/sizeof(dense[0]); k++) {
                snprintf(nm, sizeof(nm), "blk.%d.%s", L, dense[k]);
                if (!gguf_find(&m, nm)) { printf("  MISSING: %s\n", nm); ok = 0; }
            }
        } else {
            /* MoE: routed experts + shared expert + routing gate */
            static const char *moe[] = {
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
            };
            for (size_t k = 0; k < sizeof(moe)/sizeof(moe[0]); k++) {
                snprintf(nm, sizeof(nm), "blk.%d.%s", L, moe[k]);
                if (!gguf_find(&m, nm)) { printf("  MISSING: %s\n", nm); ok = 0; }
            }
            /* noaux_tc adds a per-expert bias */
            snprintf(nm, sizeof(nm), "blk.%d.exp_probs_b.bias", L);
            if (!gguf_find(&m, nm)) {
                snprintf(nm, sizeof(nm), "blk.%d.exp_probs_b.weight", L);
                if (!gguf_find(&m, nm)) {
                    printf("  NOTE: no exp_probs_b on layer %d "
                           "(group-limited router may be absent)\n", L);
                }
            }
        }
    }
    printf("  structural check: %s\n\n", ok ? "PASS" : "FAIL");

    /* --- Spot-check: pread tensor 0 header bytes to confirm fd is live --- */
    if (m.total_tensors > 0) {
        const gguf_tensor *t = m.all_tensors[0];
        uint8_t buf[16] = {0};
        int64_t r = gguf_tensor_pread(t, buf, sizeof(buf), 0);
        printf("=== I/O spot check ===\n");
        printf("  pread('%s', 16 bytes @ abs_off=%" PRIu64 ") -> %" PRId64 "\n",
               t->name, t->abs_offset, r);
        if (r > 0) {
            printf("  first bytes:");
            for (int i = 0; i < (int)r; i++) printf(" %02x", buf[i]);
            printf("\n");
        }
    }

    free(sorted);
    gguf_free_manifest(&m);
    return ok ? 0 : 2;
}
