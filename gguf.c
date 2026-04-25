/*
 * gguf.c — GGUF v3 parser + multi-shard manifest implementation.
 *
 * Format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * Design notes:
 *   - We do not mmap the GGUF files at parse time. We read the header
 *     sequentially with pread() into heap buffers. Tensor *data* is read
 *     later by consumers via pread on the kept-open fd. Host code that
 *     wants mmap can do so explicitly; the parser itself never touches
 *     tensor payload bytes.
 *   - Hash table uses FNV-1a and open addressing with linear probing.
 *   - All error paths call gguf_free_shard/manifest to avoid leaks.
 */
#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC        0x46554747u  /* "GGUF" little-endian */
#define GGUF_MAX_DIMS     4
#define GGUF_DEFAULT_ALIGN 32u

#define LOGE(...) fprintf(stderr, "[gguf] ERROR: " __VA_ARGS__)
#define LOGW(...) fprintf(stderr, "[gguf] WARN:  " __VA_ARGS__)

/* ============================================================================
 * ggml type tables. Values must match llama.cpp's ggml.c exactly.
 * ============================================================================ */
typedef struct {
    const char *name;
    size_t      blck_size;
    size_t      type_size;
} type_trait;

static const type_trait TYPE_TRAITS[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]     = { "F32",     1,   4 },
    [GGML_TYPE_F16]     = { "F16",     1,   2 },
    [GGML_TYPE_Q4_0]    = { "Q4_0",    32,  18 },
    [GGML_TYPE_Q4_1]    = { "Q4_1",    32,  20 },
    [GGML_TYPE_Q5_0]    = { "Q5_0",    32,  22 },
    [GGML_TYPE_Q5_1]    = { "Q5_1",    32,  24 },
    [GGML_TYPE_Q8_0]    = { "Q8_0",    32,  34 },
    [GGML_TYPE_Q8_1]    = { "Q8_1",    32,  36 },
    [GGML_TYPE_Q2_K]    = { "Q2_K",    256, 84 },
    [GGML_TYPE_Q3_K]    = { "Q3_K",    256, 110 },
    [GGML_TYPE_Q4_K]    = { "Q4_K",    256, 144 },
    [GGML_TYPE_Q5_K]    = { "Q5_K",    256, 176 },
    [GGML_TYPE_Q6_K]    = { "Q6_K",    256, 210 },
    [GGML_TYPE_Q8_K]    = { "Q8_K",    256, 292 },
    [GGML_TYPE_IQ2_XXS] = { "IQ2_XXS", 256, 66 },
    [GGML_TYPE_IQ2_XS]  = { "IQ2_XS",  256, 74 },
    [GGML_TYPE_IQ3_XXS] = { "IQ3_XXS", 256, 98 },
    [GGML_TYPE_IQ1_S]   = { "IQ1_S",   256, 50 },
    [GGML_TYPE_IQ4_NL]  = { "IQ4_NL",  32,  18 },
    [GGML_TYPE_IQ3_S]   = { "IQ3_S",   256, 110 },
    [GGML_TYPE_IQ2_S]   = { "IQ2_S",   256, 82 },
    [GGML_TYPE_IQ4_XS]  = { "IQ4_XS",  256, 136 },
    [GGML_TYPE_I8]      = { "I8",      1,   1 },
    [GGML_TYPE_I16]     = { "I16",     1,   2 },
    [GGML_TYPE_I32]     = { "I32",     1,   4 },
    [GGML_TYPE_I64]     = { "I64",     1,   8 },
    [GGML_TYPE_F64]     = { "F64",     1,   8 },
    [GGML_TYPE_IQ1_M]   = { "IQ1_M",   256, 56 },
    [GGML_TYPE_BF16]    = { "BF16",    1,   2 },
};

const char *ggml_type_name(ggml_type t) {
    if ((unsigned)t >= GGML_TYPE_COUNT) return "UNKNOWN";
    const char *n = TYPE_TRAITS[t].name;
    return n ? n : "UNKNOWN";
}

size_t ggml_type_size(ggml_type t) {
    if ((unsigned)t >= GGML_TYPE_COUNT) return 0;
    return TYPE_TRAITS[t].type_size;
}

size_t ggml_blck_size(ggml_type t) {
    if ((unsigned)t >= GGML_TYPE_COUNT) return 0;
    return TYPE_TRAITS[t].blck_size;
}

uint64_t gguf_tensor_nbytes(ggml_type t, const uint64_t *dims, uint32_t n_dims) {
    if ((unsigned)t >= GGML_TYPE_COUNT || n_dims == 0 || n_dims > GGUF_MAX_DIMS)
        return 0;
    size_t blck = TYPE_TRAITS[t].blck_size;
    size_t tsz  = TYPE_TRAITS[t].type_size;
    if (blck == 0 || tsz == 0) return 0;

    uint64_t nelems = 1;
    for (uint32_t i = 0; i < n_dims; i++) nelems *= dims[i];
    if (nelems % blck != 0) {
        LOGW("tensor nelems=%llu not multiple of block size %zu for %s\n",
             (unsigned long long)nelems, blck, TYPE_TRAITS[t].name);
        /* llama.cpp rounds up; we reject to fail loudly on format drift */
        return 0;
    }
    return (nelems / blck) * tsz;
}

static const char *VALUE_TYPE_NAMES[GGUF_TYPE_COUNT] = {
    "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
    "BOOL",  "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64",
};
const char *gguf_value_type_name(gguf_value_type t) {
    if ((unsigned)t >= GGUF_TYPE_COUNT) return "UNKNOWN";
    return VALUE_TYPE_NAMES[t];
}

/* ============================================================================
 * Streaming reader: we pull the header + kv + tensor-info sections into a
 * single heap buffer, then parse with a cursor. This is simple and fast
 * (headers are a few MB at most).
 * ============================================================================ */
typedef struct {
    const uint8_t *buf;
    size_t         len;
    size_t         pos;
    int            ok;
} cursor;

static int cur_read(cursor *c, void *dst, size_t n) {
    if (!c->ok) return -1;
    if (c->pos + n > c->len) { c->ok = 0; return -1; }
    memcpy(dst, c->buf + c->pos, n);
    c->pos += n;
    return 0;
}

static int cur_u8 (cursor *c, uint8_t  *v) { return cur_read(c, v, 1); }
static int cur_u16(cursor *c, uint16_t *v) { return cur_read(c, v, 2); }
static int cur_u32(cursor *c, uint32_t *v) { return cur_read(c, v, 4); }
static int cur_u64(cursor *c, uint64_t *v) { return cur_read(c, v, 8); }
static int cur_i8 (cursor *c, int8_t   *v) { return cur_read(c, v, 1); }
static int cur_i16(cursor *c, int16_t  *v) { return cur_read(c, v, 2); }
static int cur_i32(cursor *c, int32_t  *v) { return cur_read(c, v, 4); }
static int cur_i64(cursor *c, int64_t  *v) { return cur_read(c, v, 8); }
static int cur_f32(cursor *c, float    *v) { return cur_read(c, v, 4); }
static int cur_f64(cursor *c, double   *v) { return cur_read(c, v, 8); }

/* GGUF string: uint64 length, then that many bytes, not NUL-terminated.
 * We allocate a NUL-terminated copy. */
static int cur_str(cursor *c, char **out) {
    uint64_t n;
    if (cur_u64(c, &n) < 0) return -1;
    if (n > (64u << 20)) {   /* sanity cap: 64MB per string */
        LOGE("string length %llu exceeds cap\n", (unsigned long long)n);
        c->ok = 0; return -1;
    }
    if (c->pos + n > c->len) { c->ok = 0; return -1; }
    char *s = malloc(n + 1);
    if (!s) { c->ok = 0; return -1; }
    memcpy(s, c->buf + c->pos, n);
    s[n] = '\0';
    c->pos += n;
    *out = s;
    return 0;
}

/* Read a scalar GGUF value into a gguf_kv's scalar union. Returns size of
 * scalar value, or <0 on error. */
static int read_scalar(cursor *c, gguf_value_type t, gguf_kv *kv) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return cur_u8 (c, &kv->v.u8);
        case GGUF_TYPE_INT8:    return cur_i8 (c, &kv->v.i8);
        case GGUF_TYPE_UINT16:  return cur_u16(c, &kv->v.u16);
        case GGUF_TYPE_INT16:   return cur_i16(c, &kv->v.i16);
        case GGUF_TYPE_UINT32:  return cur_u32(c, &kv->v.u32);
        case GGUF_TYPE_INT32:   return cur_i32(c, &kv->v.i32);
        case GGUF_TYPE_FLOAT32: return cur_f32(c, &kv->v.f32);
        case GGUF_TYPE_UINT64:  return cur_u64(c, &kv->v.u64);
        case GGUF_TYPE_INT64:   return cur_i64(c, &kv->v.i64);
        case GGUF_TYPE_FLOAT64: return cur_f64(c, &kv->v.f64);
        case GGUF_TYPE_BOOL: {
            uint8_t b;
            if (cur_u8(c, &b) < 0) return -1;
            kv->v.b = b != 0;
            return 0;
        }
        case GGUF_TYPE_STRING:
            return cur_str(c, &kv->v.str);
        default:
            LOGE("invalid scalar type %d in read_scalar\n", (int)t);
            c->ok = 0;
            return -1;
    }
}

/* Read an array of values. For non-string arrays we slurp the raw bytes;
 * for string arrays we materialize char* pointers. */
static int read_array(cursor *c, gguf_kv *kv) {
    uint32_t elem_type_u32;
    uint64_t n;
    if (cur_u32(c, &elem_type_u32) < 0) return -1;
    if (cur_u64(c, &n) < 0) return -1;

    gguf_value_type et = (gguf_value_type)elem_type_u32;
    kv->arr.elem_type = et;
    kv->arr.n         = n;
    kv->arr.data      = NULL;

    if (n == 0) return 0;

    /* Bound: 64M elements max to avoid pathological headers */
    if (n > (64ull << 20)) {
        LOGE("array length %llu exceeds sanity cap\n", (unsigned long long)n);
        c->ok = 0; return -1;
    }

    if (et == GGUF_TYPE_STRING) {
        char **strs = calloc(n, sizeof(char*));
        if (!strs) { c->ok = 0; return -1; }
        for (uint64_t i = 0; i < n; i++) {
            if (cur_str(c, &strs[i]) < 0) {
                for (uint64_t j = 0; j < i; j++) free(strs[j]);
                free(strs);
                return -1;
            }
        }
        kv->arr.data = strs;
        return 0;
    }

    /* Fixed-size scalar arrays */
    size_t elem_sz = 0;
    switch (et) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: elem_sz = 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                    elem_sz = 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: elem_sz = 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: elem_sz = 8; break;
        default:
            LOGE("nested/unsupported array element type %d\n", (int)et);
            c->ok = 0;
            return -1;
    }

    size_t nbytes = (size_t)(n * elem_sz);
    if (c->pos + nbytes > c->len) { c->ok = 0; return -1; }
    void *data = malloc(nbytes);
    if (!data) { c->ok = 0; return -1; }
    memcpy(data, c->buf + c->pos, nbytes);
    c->pos += nbytes;
    kv->arr.data = data;
    return 0;
}

static void free_kv(gguf_kv *kv) {
    if (!kv) return;
    free(kv->key);
    if (kv->type == GGUF_TYPE_STRING) {
        free(kv->v.str);
    } else if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->arr.elem_type == GGUF_TYPE_STRING && kv->arr.data) {
            char **strs = kv->arr.data;
            for (uint64_t i = 0; i < kv->arr.n; i++) free(strs[i]);
        }
        free(kv->arr.data);
    }
    memset(kv, 0, sizeof(*kv));
}

/* ============================================================================
 * Shard loader
 * ============================================================================ */
static int read_file_range(int fd, void *buf, size_t n, off_t off) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, (char*)buf + got, n - got, off + got);
        if (r == 0) return -1;            /* unexpected EOF */
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        got += (size_t)r;
    }
    return 0;
}

int gguf_load_shard(gguf_shard *s, const char *path) {
    memset(s, 0, sizeof(*s));
    s->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { LOGE("open %s: %s\n", path, strerror(errno)); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { LOGE("fstat %s: %s\n", path, strerror(errno)); close(fd); return -1; }

    s->path      = strdup(path);
    s->fd        = fd;
    s->file_size = (uint64_t)st.st_size;
    s->alignment = GGUF_DEFAULT_ALIGN;

    /* Read a reasonable amount of header+KV+tensor-info. GGUF tensor infos
     * for 400+ GB models top out at a few MB; we grow if needed. */
    size_t hdr_cap = 8u << 20;  /* 8 MB initial */
    if (hdr_cap > s->file_size) hdr_cap = s->file_size;
    uint8_t *hdr = malloc(hdr_cap);
    if (!hdr) { close(fd); return -1; }
    if (read_file_range(fd, hdr, hdr_cap, 0) < 0) {
        LOGE("short read on header of %s\n", path);
        free(hdr); gguf_free_shard(s); return -1;
    }

retry_with_bigger_header:
    {
        cursor c = { hdr, hdr_cap, 0, 1 };

        uint32_t magic, version;
        uint64_t n_tensors, n_kv;
        if (cur_u32(&c, &magic)   < 0 ||
            cur_u32(&c, &version) < 0 ||
            cur_u64(&c, &n_tensors) < 0 ||
            cur_u64(&c, &n_kv) < 0) {
            LOGE("short header in %s\n", path);
            free(hdr); gguf_free_shard(s); return -1;
        }

        if (magic != GGUF_MAGIC) {
            LOGE("bad magic 0x%08x in %s (expected GGUF)\n", magic, path);
            free(hdr); gguf_free_shard(s); return -1;
        }
        if (version != 2 && version != 3) {
            LOGE("unsupported GGUF version %u in %s\n", version, path);
            free(hdr); gguf_free_shard(s); return -1;
        }

        s->version   = version;
        s->n_kv      = n_kv;
        s->n_tensors = n_tensors;

        /* ---- Parse KVs ---- */
        s->kvs = calloc(n_kv, sizeof(gguf_kv));
        if (!s->kvs && n_kv > 0) { free(hdr); gguf_free_shard(s); return -1; }
        for (uint64_t i = 0; i < n_kv; i++) {
            gguf_kv *kv = &s->kvs[i];
            if (cur_str(&c, &kv->key) < 0) goto need_more;

            uint32_t tu32;
            if (cur_u32(&c, &tu32) < 0) goto need_more;
            kv->type = (gguf_value_type)tu32;

            if (kv->type == GGUF_TYPE_ARRAY) {
                if (read_array(&c, kv) < 0) goto need_more;
            } else {
                if (read_scalar(&c, kv->type, kv) < 0) goto need_more;
            }

            /* Capture alignment override if present */
            if (strcmp(kv->key, "general.alignment") == 0) {
                if (kv->type == GGUF_TYPE_UINT32) s->alignment = kv->v.u32;
                else if (kv->type == GGUF_TYPE_UINT64) s->alignment = kv->v.u64;
            }
        }

        /* ---- Parse tensor infos ---- */
        s->tensors = calloc(n_tensors, sizeof(gguf_tensor));
        if (!s->tensors && n_tensors > 0) { free(hdr); gguf_free_shard(s); return -1; }
        for (uint64_t i = 0; i < n_tensors; i++) {
            gguf_tensor *t = &s->tensors[i];
            t->fd        = fd;
            t->shard_idx = -1;          /* filled in by manifest loader */

            if (cur_str(&c, &t->name) < 0) goto need_more;

            uint32_t nd;
            if (cur_u32(&c, &nd) < 0) goto need_more;
            if (nd == 0 || nd > GGUF_MAX_DIMS) {
                LOGE("tensor %s has bad n_dims=%u\n", t->name, nd);
                free(hdr); gguf_free_shard(s); return -1;
            }
            t->n_dims = nd;

            for (uint32_t d = 0; d < nd; d++) {
                if (cur_u64(&c, &t->dims[d]) < 0) goto need_more;
            }

            uint32_t tt;
            if (cur_u32(&c, &tt) < 0) goto need_more;
            t->type = (ggml_type)tt;

            if (cur_u64(&c, &t->rel_offset) < 0) goto need_more;

            t->nbytes = gguf_tensor_nbytes(t->type, t->dims, t->n_dims);
            if (t->nbytes == 0) {
                LOGE("tensor %s has zero nbytes (type=%s)\n",
                     t->name, ggml_type_name(t->type));
                free(hdr); gguf_free_shard(s); return -1;
            }
        }

        /* ---- Compute data_start: cursor position aligned up to alignment ---- */
        uint64_t after_infos = c.pos;
        uint64_t align = s->alignment ? s->alignment : GGUF_DEFAULT_ALIGN;
        s->data_start = (after_infos + align - 1) / align * align;

        for (uint64_t i = 0; i < n_tensors; i++) {
            s->tensors[i].abs_offset = s->data_start + s->tensors[i].rel_offset;
            if (s->tensors[i].abs_offset + s->tensors[i].nbytes > s->file_size) {
                LOGE("tensor %s extends past EOF in %s "
                     "(abs_off=%llu nbytes=%llu file_size=%llu)\n",
                     s->tensors[i].name, path,
                     (unsigned long long)s->tensors[i].abs_offset,
                     (unsigned long long)s->tensors[i].nbytes,
                     (unsigned long long)s->file_size);
                free(hdr); gguf_free_shard(s); return -1;
            }
        }

        free(hdr);
        return 0;

    need_more:
        /* Header buffer was too small. Double it and retry. Give up if we
         * would exceed 512 MB (paranoid ceiling). */
        if (hdr_cap >= (512u << 20)) {
            LOGE("header buffer ran out at %zu bytes on %s\n", hdr_cap, path);
            free(hdr); gguf_free_shard(s); return -1;
        }
        /* Free anything partially parsed */
        for (uint64_t i = 0; i < n_kv; i++) free_kv(&s->kvs[i]);
        free(s->kvs); s->kvs = NULL;
        if (s->tensors) {
            for (uint64_t i = 0; i < n_tensors; i++) free(s->tensors[i].name);
            free(s->tensors); s->tensors = NULL;
        }
        hdr_cap = hdr_cap * 2;
        if (hdr_cap > s->file_size) hdr_cap = s->file_size;
        uint8_t *nhdr = realloc(hdr, hdr_cap);
        if (!nhdr) { free(hdr); gguf_free_shard(s); return -1; }
        hdr = nhdr;
        if (read_file_range(fd, hdr, hdr_cap, 0) < 0) {
            free(hdr); gguf_free_shard(s); return -1;
        }
        goto retry_with_bigger_header;
    }
}

void gguf_free_shard(gguf_shard *s) {
    if (!s) return;
    if (s->fd >= 0) close(s->fd);
    free(s->path);
    if (s->kvs) {
        for (uint64_t i = 0; i < s->n_kv; i++) free_kv(&s->kvs[i]);
        free(s->kvs);
    }
    if (s->tensors) {
        for (uint64_t i = 0; i < s->n_tensors; i++) free(s->tensors[i].name);
        free(s->tensors);
    }
    memset(s, 0, sizeof(*s));
    s->fd = -1;
}

const gguf_kv *gguf_find_kv(const gguf_shard *s, const char *key) {
    if (!s || !key) return NULL;
    for (uint64_t i = 0; i < s->n_kv; i++) {
        if (strcmp(s->kvs[i].key, key) == 0) return &s->kvs[i];
    }
    return NULL;
}

int64_t gguf_tensor_pread(const gguf_tensor *t, void *buf, uint64_t n, uint64_t rel) {
    if (!t || !buf) return -1;
    if (rel + n > t->nbytes) return -1;
    ssize_t r = pread(t->fd, buf, (size_t)n, (off_t)(t->abs_offset + rel));
    if (r < 0) return -1;
    return (int64_t)r;
}

/* ============================================================================
 * Hash table (FNV-1a + linear probing)
 * ============================================================================ */
static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 16777619u; }
    return h;
}

static uint32_t next_pow2_u32(uint64_t n) {
    if (n <= 1) return 1;
    uint32_t x = 1;
    while ((uint64_t)x < n) x <<= 1;
    return x;
}

/* ============================================================================
 * Manifest loader
 * ============================================================================ */
int gguf_load_manifest(gguf_manifest *m, const char *const *paths, int n_paths) {
    memset(m, 0, sizeof(*m));
    if (n_paths <= 0) { LOGE("no shards\n"); return -1; }

    m->shards = calloc(n_paths, sizeof(gguf_shard));
    if (!m->shards) return -1;
    m->n_shards = n_paths;

    uint64_t total = 0;
    for (int i = 0; i < n_paths; i++) {
        if (gguf_load_shard(&m->shards[i], paths[i]) < 0) {
            gguf_free_manifest(m);
            return -1;
        }
        total += m->shards[i].n_tensors;
    }
    m->total_tensors = total;

    /* Flatten tensor pointers */
    m->all_tensors = calloc(total, sizeof(gguf_tensor*));
    if (!m->all_tensors) { gguf_free_manifest(m); return -1; }
    uint64_t k = 0;
    for (int i = 0; i < n_paths; i++) {
        for (uint64_t j = 0; j < m->shards[i].n_tensors; j++) {
            gguf_tensor *t = &m->shards[i].tensors[j];
            t->shard_idx = i;
            m->all_tensors[k++] = t;
        }
    }

    /* Build hash table. Size = next power of two >= total*2, min 1024. */
    uint32_t ht_size = next_pow2_u32(total * 2);
    if (ht_size < 1024) ht_size = 1024;
    m->ht = calloc(ht_size, sizeof(gguf_tensor*));
    if (!m->ht) { gguf_free_manifest(m); return -1; }
    m->ht_mask = ht_size - 1;

    for (uint64_t i = 0; i < total; i++) {
        gguf_tensor *t = m->all_tensors[i];
        uint32_t idx = fnv1a(t->name) & m->ht_mask;
        while (m->ht[idx]) {
            if (strcmp(m->ht[idx]->name, t->name) == 0) {
                LOGE("duplicate tensor name '%s' across shards (%s and %s)\n",
                     t->name,
                     m->shards[m->ht[idx]->shard_idx].path,
                     m->shards[t->shard_idx].path);
                gguf_free_manifest(m);
                return -1;
            }
            idx = (idx + 1) & m->ht_mask;
        }
        m->ht[idx] = t;
    }

    return 0;
}

void gguf_free_manifest(gguf_manifest *m) {
    if (!m) return;
    free(m->ht);
    free(m->all_tensors);
    if (m->shards) {
        for (int i = 0; i < m->n_shards; i++) gguf_free_shard(&m->shards[i]);
        free(m->shards);
    }
    memset(m, 0, sizeof(*m));
}

const gguf_tensor *gguf_find(const gguf_manifest *m, const char *name) {
    if (!m || !name || !m->ht) return NULL;
    uint32_t idx = fnv1a(name) & m->ht_mask;
    for (uint32_t probes = 0; probes <= m->ht_mask; probes++) {
        gguf_tensor *t = m->ht[idx];
        if (!t) return NULL;
        if (strcmp(t->name, name) == 0) return t;
        idx = (idx + 1) & m->ht_mask;
    }
    return NULL;
}
