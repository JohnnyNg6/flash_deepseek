/*
 * repack_experts.c — Repack DeepSeek-R1 stacked expert weights from GGUF
 * into per-layer contiguous binary files for pread()-based streaming.
 *
 * NOTE: DeepSeek-R1 Q4_K_M uses mixed quantization for down_proj:
 *   some layers have Q6_K down_proj, others have Q4_K down_proj.
 *   This repacker detects the type per layer and writes layer files
 *   whose per-expert size is:
 *     GATE (Q4_K) + UP (Q4_K) + DOWN (Q4_K or Q6_K depending on layer).
 *   infer.m must fstat() the layer file to discover the per-expert size.
 */

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <getopt.h>
#include <glob.h>

#define NUM_EXPERTS       256
#define NUM_LAYERS        61
#define FIRST_MOE_LAYER   3
#define HIDDEN_DIM        7168
#define MOE_INTERMEDIATE  2048

#define Q4K_BLOCK_ELEMS 256
#define Q4K_BLOCK_BYTES 144
#define Q6K_BLOCK_ELEMS 256
#define Q6K_BLOCK_BYTES 210

/* gate_proj/up_proj are always Q4_K in Q4_K_M for DeepSeek-R1 */
#define GATE_PROJ_BYTES  (57344ULL * Q4K_BLOCK_BYTES)   /* 8,257,536 */
#define UP_PROJ_BYTES    (57344ULL * Q4K_BLOCK_BYTES)   /* 8,257,536 */

/* down_proj size depends on quantization type per layer */
#define DOWN_PROJ_Q4K_BYTES (57344ULL * Q4K_BLOCK_BYTES) /* 8,257,536 */
#define DOWN_PROJ_Q6K_BYTES (57344ULL * Q6K_BLOCK_BYTES) /* 12,042,240 */

static const char *DEFAULT_MODEL_PATH =
    "/Users/user11/models/deepseek-r1-q4km/DeepSeek-R1-Q4_K_M";

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL_PATH;
    int single_layer = -1;

    static struct option opts[] = {
        {"model", required_argument, 0, 'm'},
        {"layer", required_argument, 0, 'l'},
        {"help",  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "m:l:h", opts, NULL)) != -1) {
        switch (c) {
            case 'm': model_path = optarg; break;
            case 'l': single_layer = atoi(optarg); break;
            case 'h':
                printf("Usage: %s [--model PATH] [--layer N]\n", argv[0]);
                return 0;
            default: return 1;
        }
    }

    printf("=== DeepSeek-R1 Expert Repacker ===\n");
    printf("Model: %s\n", model_path);

    /* Find all GGUF shards */
    char pattern[2048];
    snprintf(pattern, sizeof(pattern), "%s/DeepSeek-R1-Q4_K_M-*-of-*.gguf", model_path);

    glob_t gl;
    if (glob(pattern, GLOB_NOSORT, NULL, &gl) != 0 || gl.gl_pathc == 0) {
        fprintf(stderr, "ERROR: No GGUF shards found matching %s\n", pattern);
        return 1;
    }

    int n_shards = (int)gl.gl_pathc;
    const char **shard_paths = malloc(n_shards * sizeof(char *));
    for (int i = 0; i < n_shards; i++) shard_paths[i] = gl.gl_pathv[i];
    for (int i = 0; i < n_shards - 1; i++)
        for (int j = i + 1; j < n_shards; j++)
            if (strcmp(shard_paths[i], shard_paths[j]) > 0) {
                const char *tmp = shard_paths[i];
                shard_paths[i] = shard_paths[j];
                shard_paths[j] = tmp;
            }

    printf("Found %d GGUF shards\n", n_shards);

    gguf_manifest manifest;
    if (gguf_load_manifest(&manifest, shard_paths, n_shards) != 0) {
        fprintf(stderr, "ERROR: Failed to load GGUF manifest\n");
        return 1;
    }
    printf("Loaded manifest: %llu tensors\n",
           (unsigned long long)manifest.total_tensors);

    char out_dir[2048];
    snprintf(out_dir, sizeof(out_dir), "%s/packed_experts", model_path);
    mkdir(out_dir, 0755);

    int first_layer = FIRST_MOE_LAYER;
    int last_layer = NUM_LAYERS - 1;
    if (single_layer >= 0) {
        first_layer = single_layer;
        last_layer = single_layer;
    }

    double t_total_start = now_ms();
    uint64_t total_bytes_written = 0;
    int n_q4_down = 0, n_q6_down = 0;

    for (int L = first_layer; L <= last_layer; L++) {
        char name_gate[128], name_up[128], name_down[128];
        snprintf(name_gate, sizeof(name_gate), "blk.%d.ffn_gate_exps.weight", L);
        snprintf(name_up,   sizeof(name_up),   "blk.%d.ffn_up_exps.weight",   L);
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down_exps.weight", L);

        const gguf_tensor *t_gate = gguf_find(&manifest, name_gate);
        const gguf_tensor *t_up   = gguf_find(&manifest, name_up);
        const gguf_tensor *t_down = gguf_find(&manifest, name_down);

        if (!t_gate || !t_up || !t_down) {
            if (L < FIRST_MOE_LAYER) continue;
            fprintf(stderr, "WARNING: Layer %d missing expert tensors, skipping\n", L);
            continue;
        }

        /* Determine per-expert down size from actual tensor type */
        uint64_t down_proj_bytes;
        const char *down_type_str;
        if (t_down->type == GGML_TYPE_Q6_K) {
            down_proj_bytes = DOWN_PROJ_Q6K_BYTES;
            down_type_str = "Q6_K";
            n_q6_down++;
        } else if (t_down->type == GGML_TYPE_Q4_K) {
            down_proj_bytes = DOWN_PROJ_Q4K_BYTES;
            down_type_str = "Q4_K";
            n_q4_down++;
        } else {
            fprintf(stderr, "ERROR: Layer %d down_proj has unsupported type %s, skipping\n",
                    L, ggml_type_name(t_down->type));
            continue;
        }

        uint64_t gate_off   = 0;
        uint64_t up_off     = GATE_PROJ_BYTES;
        uint64_t down_off   = GATE_PROJ_BYTES + UP_PROJ_BYTES;
        uint64_t expert_size = GATE_PROJ_BYTES + UP_PROJ_BYTES + down_proj_bytes;

        /* Verify sizes */
        uint64_t expected_gate_total = (uint64_t)NUM_EXPERTS * GATE_PROJ_BYTES;
        uint64_t expected_up_total   = (uint64_t)NUM_EXPERTS * UP_PROJ_BYTES;
        uint64_t expected_down_total = (uint64_t)NUM_EXPERTS * down_proj_bytes;

        if (t_gate->nbytes != expected_gate_total ||
            t_up->nbytes   != expected_up_total ||
            t_down->nbytes != expected_down_total) {
            fprintf(stderr, "ERROR: Layer %d tensor sizes don't match expected:\n", L);
            fprintf(stderr, "  gate: %llu (expected %llu)\n",
                    (unsigned long long)t_gate->nbytes,
                    (unsigned long long)expected_gate_total);
            fprintf(stderr, "  up:   %llu (expected %llu)\n",
                    (unsigned long long)t_up->nbytes,
                    (unsigned long long)expected_up_total);
            fprintf(stderr, "  down: %llu (expected %llu, type=%s)\n",
                    (unsigned long long)t_down->nbytes,
                    (unsigned long long)expected_down_total,
                    down_type_str);
            continue;
        }

        char out_path[2048];
        snprintf(out_path, sizeof(out_path), "%s/layer_%02d.bin", out_dir, L);
        int fd_out = open(out_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd_out < 0) {
            fprintf(stderr, "ERROR: Cannot create %s: %s\n", out_path, strerror(errno));
            continue;
        }

        uint64_t layer_size = (uint64_t)NUM_EXPERTS * expert_size;
        ftruncate(fd_out, layer_size);

        double t_layer = now_ms();
        printf("Layer %2d: down=%s, expert_size=%llu B (%.2f MB), repacking %d experts (%.2f GB) ...\n",
               L, down_type_str,
               (unsigned long long)expert_size,
               expert_size / (1024.0 * 1024.0),
               NUM_EXPERTS,
               layer_size / (1024.0 * 1024.0 * 1024.0));

        void *buf = malloc(expert_size);
        if (!buf) { close(fd_out); continue; }

        for (int e = 0; e < NUM_EXPERTS; e++) {
            uint64_t src_gate_off = (uint64_t)e * GATE_PROJ_BYTES;
            int64_t r = gguf_tensor_pread(t_gate, (char *)buf + gate_off,
                                          GATE_PROJ_BYTES, src_gate_off);
            if (r != (int64_t)GATE_PROJ_BYTES) {
                fprintf(stderr, "ERROR: layer %d expert %d gate pread: %lld\n",
                        L, e, (long long)r);
                break;
            }

            uint64_t src_up_off = (uint64_t)e * UP_PROJ_BYTES;
            r = gguf_tensor_pread(t_up, (char *)buf + up_off,
                                  UP_PROJ_BYTES, src_up_off);
            if (r != (int64_t)UP_PROJ_BYTES) {
                fprintf(stderr, "ERROR: layer %d expert %d up pread: %lld\n",
                        L, e, (long long)r);
                break;
            }

            uint64_t src_down_off = (uint64_t)e * down_proj_bytes;
            r = gguf_tensor_pread(t_down, (char *)buf + down_off,
                                  down_proj_bytes, src_down_off);
            if (r != (int64_t)down_proj_bytes) {
                fprintf(stderr, "ERROR: layer %d expert %d down pread: %lld\n",
                        L, e, (long long)r);
                break;
            }

            off_t expert_file_off = (off_t)e * expert_size;
            ssize_t w = pwrite(fd_out, buf, expert_size, expert_file_off);
            if (w != (ssize_t)expert_size) {
                fprintf(stderr, "ERROR: layer %d expert %d write: %zd\n", L, e, w);
                break;
            }

            if ((e + 1) % 64 == 0 || e == NUM_EXPERTS - 1) {
                double elapsed = (now_ms() - t_layer) / 1000.0;
                double rate = (e + 1) / elapsed;
                double eta = (NUM_EXPERTS - e - 1) / rate;
                printf("  [%3d/%d] %.1fs elapsed, %.0f experts/s, ETA %.0fs\n",
                       e + 1, NUM_EXPERTS, elapsed, rate, eta);
            }
        }

        free(buf);
        close(fd_out);

        double layer_ms = now_ms() - t_layer;
        printf("  Layer %d done: %.1fs (%.1f GB/s)\n",
               L, layer_ms / 1000.0,
               layer_size / (layer_ms * 1e6));
        total_bytes_written += layer_size;
    }

    double total_s = (now_ms() - t_total_start) / 1000.0;
    printf("\n=== Done ===\n");
    printf("Layers with Q6_K down: %d\n", n_q6_down);
    printf("Layers with Q4_K down: %d\n", n_q4_down);
    printf("Total: %.1f GB in %.0fs (%.1f GB/s)\n",
           total_bytes_written / 1e9, total_s,
           total_bytes_written / (total_s * 1e9));
    printf("Output: %s/\n", out_dir);

    free(shard_paths);
    globfree(&gl);
    gguf_free_manifest(&manifest);
    return 0;
}
