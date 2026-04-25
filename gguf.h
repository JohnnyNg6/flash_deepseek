/*
 * gguf.h — GGUF v3 parser + multi-shard tensor manifest (Phase A1).
 *
 * Parses the GGUF file format (llama.cpp) into an in-memory manifest.
 * Supports multi-shard files (e.g. DeepSeek-R1 split across 11 GGUFs).
 * Each shard is opened, its header parsed, and its tensors registered
 * into a single hash table keyed by tensor name.
 *
 * No dynamic allocation during the lookup hot path; the hash table is
 * sized once at manifest load time.
 */
#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GGUF metadata value types (per gguf.md spec)
 * ============================================================================ */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
} gguf_value_type;

/* ============================================================================
 * GGML tensor quantization types (subset we actually care about; enum values
 * must match ggml.h exactly or tensor offsets will be misinterpreted).
 * ============================================================================ */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    /* 4, 5 removed in llama.cpp */
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT,
} ggml_type;

/* ============================================================================
 * Metadata key-value pair. String data and array data are owned by the
 * shard they belong to and freed by gguf_free_shard.
 * ============================================================================ */
typedef struct gguf_kv {
    char           *key;        /* heap-owned, NUL-terminated */
    gguf_value_type type;

    /* Scalar payload */
    union {
        uint8_t  u8;   int8_t  i8;
        uint16_t u16;  int16_t i16;
        uint32_t u32;  int32_t i32;
        uint64_t u64;  int64_t i64;
        float    f32;  double  f64;
        bool     b;
        char    *str;              /* heap-owned, NUL-terminated */
    } v;

    /* Array payload (used iff type == GGUF_TYPE_ARRAY) */
    struct {
        gguf_value_type elem_type;
        uint64_t        n;         /* number of elements */
        void           *data;      /* for strings: char**, else raw elems */
    } arr;
} gguf_kv;

/* ============================================================================
 * Tensor descriptor. Offset is absolute within the shard file: reading
 * tensor data is `pread(shard->fd, buf, t->nbytes, t->abs_offset)`.
 * ============================================================================ */
typedef struct gguf_tensor {
    char     *name;               /* heap-owned, NUL-terminated */
    uint32_t  n_dims;
    uint64_t  dims[4];            /* GGUF_MAX_DIMS = 4 */
    ggml_type type;
    uint64_t  rel_offset;         /* offset relative to shard->data_start */
    uint64_t  abs_offset;         /* = shard->data_start + rel_offset */
    uint64_t  nbytes;             /* computed from type + dims */
    int       shard_idx;          /* 0..n_shards-1 */
    int       fd;                 /* duplicated from shard for convenience */
} gguf_tensor;

/* ============================================================================
 * Per-shard context. One struct per GGUF file.
 * ============================================================================ */
typedef struct gguf_shard {
    char       *path;             /* heap-owned */
    int         fd;               /* O_RDONLY, kept open for the lifetime */
    uint32_t    version;          /* GGUF version, must be 2 or 3 */
    uint64_t    n_tensors;
    uint64_t    n_kv;
    uint64_t    alignment;        /* from general.alignment, default 32 */
    uint64_t    data_start;       /* absolute file offset where tensor data begins */
    uint64_t    file_size;
    gguf_kv    *kvs;              /* [n_kv] */
    gguf_tensor *tensors;         /* [n_tensors] */
} gguf_shard;

/* ============================================================================
 * Multi-shard manifest. Owns its shards and maintains a flat index of all
 * tensors across all shards, plus a hash table for O(1) name lookup.
 * ============================================================================ */
typedef struct gguf_manifest {
    gguf_shard     *shards;
    int             n_shards;

    /* Flat view: all_tensors[i] points into shards[shard_idx].tensors[...] */
    gguf_tensor   **all_tensors;
    uint64_t        total_tensors;

    /* Open-addressing hash table. Size is a power of two. Empty slots hold
     * NULL. Load factor kept below ~0.7. */
    uint32_t        ht_mask;      /* ht size - 1 */
    gguf_tensor   **ht;
} gguf_manifest;

/* ============================================================================
 * Public API
 * ============================================================================ */

/* Load a single shard. Returns 0 on success, <0 on error. */
int  gguf_load_shard(gguf_shard *s, const char *path);
void gguf_free_shard(gguf_shard *s);

/* Load n_paths shards into a unified manifest. Duplicate tensor names across
 * shards are a hard error. Returns 0 on success. */
int  gguf_load_manifest(gguf_manifest *m, const char *const *paths, int n_paths);
void gguf_free_manifest(gguf_manifest *m);

/* O(1) tensor lookup by name. Returns NULL if not found. */
const gguf_tensor *gguf_find(const gguf_manifest *m, const char *name);

/* O(n_kv) metadata lookup within a single shard. The model-wide KVs are
 * typically only in shard 0; check there first. */
const gguf_kv *gguf_find_kv(const gguf_shard *s, const char *key);

/* ggml type inspection */
const char *ggml_type_name(ggml_type t);
size_t      ggml_type_size(ggml_type t);   /* bytes per block */
size_t      ggml_blck_size(ggml_type t);   /* elements per block */

/* Compute tensor byte size from type + shape. Returns 0 on invalid input. */
uint64_t    gguf_tensor_nbytes(ggml_type t, const uint64_t *dims, uint32_t n_dims);

/* Value-type name (for diagnostics) */
const char *gguf_value_type_name(gguf_value_type t);

/* Read `n` bytes from `t` into `buf`. Thin wrapper around pread. Returns
 * number of bytes read on success, or <0 on error. `rel` is an offset
 * within the tensor (0 = tensor start). */
int64_t     gguf_tensor_pread(const gguf_tensor *t, void *buf,
                              uint64_t n, uint64_t rel);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_H */
