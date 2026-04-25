#include "gguf.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usage: ./quant_test <shard.gguf> <tensor_name> <n_elements_to_print>
 *
 * Prints the header line on STDERR and the first n dequantized fp32 values
 * on STDOUT, one per line in "%.6f" format — so `diff` against
 * gen_reference.py's stdout is a clean byte-for-byte comparison. */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s shard.gguf tensor_name n\n", argv[0]);
        return 1;
    }
    const char  *path   = argv[1];
    const char  *name   = argv[2];
    const size_t nprint = (size_t)strtoull(argv[3], NULL, 10);

    gguf_manifest m;
    const char *paths[1] = { path };
    if (gguf_load_manifest(&m, paths, 1) != 0) {
        fprintf(stderr, "open failed: %s\n", path);
        return 1;
    }

    const gguf_tensor *t = gguf_find(&m, name);
    if (!t) {
        fprintf(stderr, "tensor not found: %s\n", name);
        gguf_free_manifest(&m);
        return 1;
    }

    uint64_t n_elem = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) n_elem *= t->dims[i];

    void *blob = malloc(t->nbytes);
    if (!blob) {
        fprintf(stderr, "malloc(%llu) failed\n", (unsigned long long)t->nbytes);
        gguf_free_manifest(&m);
        return 1;
    }

    int64_t got = gguf_tensor_pread(t, blob, t->nbytes, 0);
    if (got < 0 || (uint64_t)got != t->nbytes) {
        fprintf(stderr, "pread failed: got=%lld expected=%llu\n",
                (long long)got, (unsigned long long)t->nbytes);
        free(blob); gguf_free_manifest(&m);
        return 1;
    }

    float *out = malloc(n_elem * sizeof(float));
    if (!out) {
        fprintf(stderr, "malloc(%llu floats) failed\n",
                (unsigned long long)n_elem);
        free(blob); gguf_free_manifest(&m);
        return 1;
    }

    const char *type_str = NULL;
    if (t->type == GGML_TYPE_Q4_K) {
        dequantize_row_q4_K(blob, out, n_elem);
        type_str = "Q4_K";
    } else if (t->type == GGML_TYPE_Q6_K) {
        dequantize_row_q6_K(blob, out, n_elem);
        type_str = "Q6_K";
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(out, blob, n_elem * sizeof(float));
        type_str = "F32";
    } else {
        fprintf(stderr, "unsupported type %d (%s)\n",
                (int)t->type, ggml_type_name(t->type));
        free(blob); free(out); gguf_free_manifest(&m);
        return 1;
    }

    /* Header -> stderr, values -> stdout. Matches gen_reference.py. */
    fprintf(stderr, "tensor %s  n_elem=%llu  type=%s  nbytes=%llu\n",
            name,
            (unsigned long long)n_elem,
            type_str,
            (unsigned long long)t->nbytes);

    for (size_t i = 0; i < nprint && i < n_elem; ++i) {
        printf("%.6f\n", out[i]);
    }

    free(blob);
    free(out);
    gguf_free_manifest(&m);
    return 0;
}
