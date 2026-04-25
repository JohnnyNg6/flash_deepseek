#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---- GPT-2 byte<->unicode map (used by DeepSeek-R1 ByteLevel pretokenizer) ----
static void build_byte_map(Tokenizer *t) {
    // The 188 "printable" bytes map to themselves; the rest map to 256+i.
    int taken[256] = {0};
    for (int b = '!'; b <= '~'; b++) { t->b2u[b] = b; taken[b] = 1; }
    for (int b = 0xA1; b <= 0xAC; b++){ t->b2u[b] = b; taken[b] = 1; }
    for (int b = 0xAE; b <= 0xFF; b++){ t->b2u[b] = b; taken[b] = 1; }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!taken[b]) { t->b2u[b] = 256 + n; n++; }
    }
    for (int i = 0; i < 512; i++) t->u2b[i] = -1;
    for (int b = 0; b < 256; b++) t->u2b[t->b2u[b]] = b;
}

// ---- FNV-1a string hash ----
static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

static uint32_t next_pow2(uint32_t x) {
    uint32_t p = 1; while (p < x) p <<= 1; return p;
}

// ---- Encode UTF-8 codepoint to UTF-8 bytes ----
static int utf8_encode(int cp, char *out) {
    if (cp < 0x80)        { out[0]=cp; return 1; }
    if (cp < 0x800)       { out[0]=0xC0|(cp>>6); out[1]=0x80|(cp&63); return 2; }
    if (cp < 0x10000)     { out[0]=0xE0|(cp>>12); out[1]=0x80|((cp>>6)&63);
                            out[2]=0x80|(cp&63); return 3; }
    out[0]=0xF0|(cp>>18); out[1]=0x80|((cp>>12)&63);
    out[2]=0x80|((cp>>6)&63); out[3]=0x80|(cp&63); return 4;
}

// ---- Decode one UTF-8 codepoint ----
static int utf8_decode(const char *s, int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80)        { *cp = c; return 1; }
    if ((c & 0xE0)==0xC0){ *cp = ((c&0x1F)<<6)|(s[1]&63); return 2; }
    if ((c & 0xF0)==0xE0){ *cp = ((c&0x0F)<<12)|((s[1]&63)<<6)|(s[2]&63); return 3; }
    *cp = ((c&0x07)<<18)|((s[1]&63)<<12)|((s[2]&63)<<6)|(s[3]&63); return 4;
}

// ---- Vocab hash insertion ----
static void ht_insert(Tokenizer *t, const char *key, int len, uint32_t vidx) {
    uint32_t h = fnv1a(key, len) & t->ht_mask;
    while (t->ht[h] != UINT32_MAX) h = (h+1) & t->ht_mask;
    t->ht[h] = vidx;
    t->ht_keys[vidx] = (char*)t->vocab[vidx].bytes; // alias
}

static int32_t ht_find(const Tokenizer *t, const char *key, int len) {
    uint32_t h = fnv1a(key, len) & t->ht_mask;
    for (uint32_t p = 0; p <= t->ht_mask; p++) {
        uint32_t v = t->ht[h];
        if (v == UINT32_MAX) return -1;
        if ((int)t->vocab[v].len == len &&
            memcmp(t->vocab[v].bytes, key, len) == 0) return (int32_t)v;
        h = (h+1) & t->ht_mask;
    }
    return -1;
}

// ---- Merge hash ----
static uint64_t pair_key(uint32_t a, uint32_t b) {
    return ((uint64_t)a << 32) | b;
}
static int merge_lookup(const Tokenizer *t, uint32_t a, uint32_t b,
                        uint32_t *rank, uint32_t *merged) {
    uint64_t k = pair_key(a, b);
    uint32_t h = (uint32_t)((k * 11400714819323198485ULL) >> 32) & t->mht_mask;
    for (uint32_t p = 0; p <= t->mht_mask; p++) {
        if (t->mht_rank[h] < 0) return 0;
        if (t->mht_keys[h] == k) {
            *rank   = (uint32_t)t->mht_rank[h];
            *merged = t->mht_merged[h];
            return 1;
        }
        h = (h+1) & t->mht_mask;
    }
    return 0;
}



int tokenizer_load(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof(*t));
    build_byte_map(t);

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return -1; }

    char magic[4];
    if (fread(magic,1,4,f)!=4 || memcmp(magic,"BPET",4)) {
        fprintf(stderr,"tokenizer: bad magic\n"); fclose(f); return -1;
    }
    uint32_t version, n_vocab, n_merges, n_added;
    fread(&version,4,1,f);
    fread(&n_vocab,4,1,f);
    fread(&n_merges,4,1,f);
    fread(&n_added,4,1,f);

    /* CRITICAL FIX: 預先配置 base + added 兩段空間。
     * 原版只配置 n_vocab 個 entry，把 added tokens (1280 個 R1 特殊 token，
     * 包含 <think>=128798, </think>=128799, <｜User｜>=128803, <｜Assistant｜>=128804
     * 等) 全部丟掉。後果是：
     *   - decode_one 對 id ≥ 128000 的 binary search 直接 miss → 顯示空字串，
     *     使用者看到模型「跳過」了 </think>，誤以為模型壞了。
     *   - chat template 雖然把 think_id 帶進 prompt，但 prompt 後面其實少了
     *     <｜Assistant｜><think>\n 結構，模型在 thinking↔answer 之間來回亂跳。
     */
    uint32_t total_cap = n_vocab + n_added;
    t->vocab = calloc(total_cap, sizeof(tk_vocab_entry));

    /* 1) Base vocab（位置 0..n_vocab-1），輸出檔已按 id 升冪寫入 */
    for (uint32_t i = 0; i < n_vocab; i++) {
        uint32_t id; uint16_t len;
        fread(&id,4,1,f); fread(&len,2,1,f);
        char *b = malloc(len+1);
        fread(b,1,len,f); b[len]=0;
        t->vocab[i].bytes = b;
        t->vocab[i].len   = len;
        t->vocab[i].id    = id;
    }

    /* 2) Hash table 大小要覆蓋 base + added，否則之後塞 added 會撞到 mask 邊界 */
    uint32_t hsz = next_pow2(total_cap * 2);
    t->ht_mask = hsz - 1;
    t->ht = malloc(hsz * 4);
    t->ht_keys = calloc(total_cap, sizeof(char*));
    for (uint32_t i = 0; i < hsz; i++) t->ht[i] = UINT32_MAX;
    for (uint32_t i = 0; i < n_vocab; i++)
        ht_insert(t, t->vocab[i].bytes, t->vocab[i].len, i);

    /* 3) Merges：使用 base vocab 的 hash 解析（merges 不會引用 added tokens） */
    uint32_t mhsz = next_pow2(n_merges * 2);
    t->mht_mask = mhsz - 1;
    t->mht_keys   = calloc(mhsz, 8);
    t->mht_rank   = malloc(mhsz * 4);
    t->mht_merged = calloc(mhsz, 4);
    for (uint32_t i = 0; i < mhsz; i++) t->mht_rank[i] = -1;

    char ab[4096];
    for (uint32_t i = 0; i < n_merges; i++) {
        uint16_t la, lb;
        fread(&la,2,1,f);
        char a[2048]; fread(a,1,la,f); a[la]=0;
        fread(&lb,2,1,f);
        char b[2048]; fread(b,1,lb,f); b[lb]=0;

        int32_t ai = ht_find(t, a, la);
        int32_t bi = ht_find(t, b, lb);
        if (ai < 0 || bi < 0) continue;

        memcpy(ab, a, la); memcpy(ab+la, b, lb);
        int32_t mi = ht_find(t, ab, la+lb);
        if (mi < 0) continue;

        uint64_t k = pair_key((uint32_t)ai, (uint32_t)bi);
        uint32_t h = (uint32_t)((k * 11400714819323198485ULL) >> 32) & t->mht_mask;
        while (t->mht_rank[h] >= 0) h = (h+1) & t->mht_mask;
        t->mht_keys[h]   = k;
        t->mht_rank[h]   = (int32_t)i;
        t->mht_merged[h] = (uint32_t)mi;
    }

    /* 4) Added tokens — 真正塞入 vocab + hash，不再丟掉。
     *    DeepSeek-R1 的 added token id 是連續的 128000..129279，比 base
     *    最大 id (127999) 大，所以「base 在前 + added 在後」的順序天然
     *    保持 id 升冪，decode_one 的 binary search 仍然有效。 */
    uint32_t added_count = 0;
    for (uint32_t i = 0; i < n_added; i++) {
        uint32_t id; uint16_t len;
        fread(&id,4,1,f); fread(&len,2,1,f);
        char *b = malloc(len+1);
        fread(b,1,len,f); b[len]=0;

        if      (strcmp(b,"<｜begin▁of▁sentence｜>")==0) t->bos_id       = id;
        else if (strcmp(b,"<｜end▁of▁sentence｜>")==0)   t->eos_id       = id;
        else if (strcmp(b,"<｜User｜>")==0)               t->user_id      = id;
        else if (strcmp(b,"<｜Assistant｜>")==0)          t->assistant_id = id;
        else if (strcmp(b,"<think>")==0)                  t->think_id     = id;

        /* 若這個 byte-form 已經在 base vocab 裡 (極少數巧合，例如純 ASCII
         * 短 token)，覆蓋它的 id；否則 append 一筆新的 vocab entry。 */
        int32_t existing = ht_find(t, b, len);
        if (existing >= 0) {
            t->vocab[existing].id = id;
            free(b);
        } else {
            uint32_t idx = n_vocab + added_count;
            t->vocab[idx].bytes = b;
            t->vocab[idx].len   = (uint16_t)len;
            t->vocab[idx].id    = id;
            ht_insert(t, b, len, idx);
            added_count++;
        }
    }
    t->n_vocab = n_vocab + added_count;
    fclose(f);

    fprintf(stderr,
        "[tokenizer] loaded %u vocab (%u base + %u added), %u merges; "
        "bos=%u eos=%u user=%u assistant=%u think=%u\n",
        t->n_vocab, n_vocab, added_count, n_merges,
        t->bos_id, t->eos_id, t->user_id, t->assistant_id, t->think_id);
    return 0;
}




void tokenizer_free(Tokenizer *t) {
    for (uint32_t i = 0; i < t->n_vocab; i++) free(t->vocab[i].bytes);
    free(t->vocab); free(t->ht); free(t->ht_keys);
    free(t->mht_keys); free(t->mht_rank); free(t->mht_merged);
}

// --- Encode: byte-level pre-tokenize then merge greedily by lowest rank ---
int tokenizer_encode(const Tokenizer *t, const char *text,
                     uint32_t *out_ids, int max_ids)
{
    // 1. Convert UTF-8 text -> byte-level unicode string (each input byte
    //    becomes one codepoint via b2u), then re-encode that codepoint as UTF-8
    //    so we can index into vocab (which stores byte-form strings).
    int n = strlen(text);
    char *buf = malloc(n * 4 + 8);
    int blen = 0;
    for (int i = 0; i < n; i++) {
        int cp = t->b2u[(unsigned char)text[i]];
        blen += utf8_encode(cp, buf + blen);
    }
    buf[blen] = 0;

    // 2. Initialize symbols: each unicode codepoint = one initial token
    int nsym = 0;
    int *sym_off = malloc(blen * sizeof(int));
    int *sym_len = malloc(blen * sizeof(int));
    int *sym_id  = malloc(blen * sizeof(int));
    int *prev    = malloc(blen * sizeof(int));
    int *next    = malloc(blen * sizeof(int));

    int p = 0;
    while (p < blen) {
        int cp, w = utf8_decode(buf + p, &cp);
        sym_off[nsym] = p;
        sym_len[nsym] = w;
        int32_t vi = ht_find(t, buf + p, w);
        sym_id[nsym] = vi >= 0 ? (int)t->vocab[vi].id : 0;
        prev[nsym] = nsym - 1;
        next[nsym] = nsym + 1;
        nsym++;
        p += w;
    }
    next[nsym - 1] = -1;

    // 3. Repeatedly find the pair with lowest merge rank and merge it.
    //    O(n^2) but fine for typical prompts.
    while (1) {
        int best_rank = INT32_MAX;
        int best_i = -1;
        uint32_t best_merged = 0;

        for (int i = 0; i < nsym && next[i] != -1; i = next[i]) {
            int j = next[i];
            if (j < 0) break;
            // Look up (vocab_index_of_a, vocab_index_of_b)
            // We need vocab indices, not ids. Re-find via byte form.
            int32_t ai = ht_find(t, buf + sym_off[i], sym_len[i]);
            int32_t bi = ht_find(t, buf + sym_off[j], sym_len[j]);
            if (ai < 0 || bi < 0) continue;
            uint32_t r, m;
            if (merge_lookup(t, (uint32_t)ai, (uint32_t)bi, &r, &m)) {
                if ((int)r < best_rank) {
                    best_rank = r;
                    best_i = i;
                    best_merged = m;
                }
            }
        }
        if (best_i < 0) break;

        // Merge symbol at best_i with next[best_i]
        int j = next[best_i];
        sym_len[best_i] += sym_len[j];
        sym_id[best_i]  = (int)t->vocab[best_merged].id;
        next[best_i] = next[j];
        if (next[j] != -1) prev[next[j]] = best_i;
    }

    // 4. Walk the linked list and emit ids
    int out_n = 0;
    for (int i = 0; i != -1 && out_n < max_ids; i = next[i]) {
        out_ids[out_n++] = (uint32_t)sym_id[i];
        if (next[i] == -1) break;
    }

    free(buf); free(sym_off); free(sym_len); free(sym_id); free(prev); free(next);
    return out_n;
}

int tokenizer_decode_one(const Tokenizer *t, uint32_t id,
                         char *out_buf, int max)
{
    // Find vocab entry by id (binary search since vocab is id-sorted)
    int lo = 0, hi = (int)t->n_vocab - 1, vi = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (t->vocab[mid].id == id) { vi = mid; break; }
        else if (t->vocab[mid].id < id) lo = mid + 1;
        else hi = mid - 1;
    }
    if (vi < 0) return 0;

    // The vocab entry stores byte-level-unicode UTF-8.
    // Decode codepoints, map back to bytes, write raw UTF-8.
    const char *s = t->vocab[vi].bytes;
    int n = t->vocab[vi].len;
    int p = 0, w = 0;
    while (p < n && w < max) {
        int cp, k = utf8_decode(s + p, &cp);
        p += k;
        if (cp < 512 && t->u2b[cp] >= 0) {
            out_buf[w++] = (char)t->u2b[cp];
        } else {
            // Special token (e.g. <｜...｜>) — pass UTF-8 through
            for (int i = 0; i < k && w < max; i++) out_buf[w++] = s[p - k + i];
        }
    }
    return w;
}



int tokenizer_apply_chat_template(const Tokenizer *t, const char *user_text,
                                  uint32_t *out_ids, int max_ids)
{
    int n = 0;

    /* BOS — DeepSeek-R1 的 BOS id 是 0（token "<｜begin▁of▁sentence｜>"），
     * 這是真實的 token id，不是「缺值」哨兵。原本的 `if (t->bos_id)` 在
     * R1 上永遠跳過 BOS，光這一條就足以讓模型進入思考迴圈、永遠走不到
     * 真正的回答。 */
    if (n < max_ids) out_ids[n++] = t->bos_id;

    if (t->user_id && n < max_ids) out_ids[n++] = t->user_id;
    n += tokenizer_encode(t, user_text, out_ids + n, max_ids - n);
    if (t->assistant_id && n < max_ids) out_ids[n++] = t->assistant_id;

    /* "<think>\n" — R1 進入 thinking mode 的必要尾巴。
     * think_id 由 tokenizer_load 中 `<think>` 的 added-token 比對設置；
     * R1 應為 128798。如果 tokenizer.bin 是舊版沒帶 <think>，退回 byte-
     * level 編碼（會被切成 <, th, ink, > 4 個 token，模型品質會略差但
     * 至少不會段錯）。 */
    if (t->think_id && n < max_ids) {
        out_ids[n++] = t->think_id;
    } else {
        n += tokenizer_encode(t, "<think>", out_ids + n, max_ids - n);
    }
    n += tokenizer_encode(t, "\n", out_ids + n, max_ids - n);

    return n;
}
