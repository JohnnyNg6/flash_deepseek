#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
    char    *bytes;     // UTF-8 byte-level encoded form (for matching merges)
    uint32_t id;
    uint16_t len;
} tk_vocab_entry;

typedef struct {
    uint32_t a_idx;     // index into vocab (by byte form)
    uint32_t b_idx;
    uint32_t merged_idx;
    uint32_t rank;      // merge rank (priority = lower)
} tk_merge;

typedef struct {
    tk_vocab_entry *vocab;       // sorted by id
    uint32_t        n_vocab;

    // Hash: byte-form string -> vocab index
    uint32_t       *ht;          // open-addressing
    uint32_t        ht_mask;
    char          **ht_keys;     // owned

    // Merges as id-pair -> rank
    // Use hash on (a_id<<32)|b_id
    uint64_t       *mht_keys;
    int32_t        *mht_rank;
    uint32_t       *mht_merged;
    uint32_t        mht_mask;

    // Special tokens
    uint32_t bos_id, eos_id;
    uint32_t user_id, assistant_id, think_id;

    // byte<->unicode map (GPT-2 style)
    int      b2u[256];           // byte -> unicode codepoint
    int      u2b[512];           // codepoint -> byte (-1 if none)
} Tokenizer;

int  tokenizer_load(Tokenizer *t, const char *path);
void tokenizer_free(Tokenizer *t);

// Encode raw UTF-8 text into token ids. Returns count, fills out_ids[0..count].
int  tokenizer_encode(const Tokenizer *t, const char *text,
                      uint32_t *out_ids, int max_ids);

// Decode a single token id into UTF-8 bytes; appends into out_buf
// (returns number of bytes written; out_buf must be >= 256).
int  tokenizer_decode_one(const Tokenizer *t, uint32_t id,
                          char *out_buf, int max);

// Apply DeepSeek-R1 chat template:
//   <｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜><think>\n
int  tokenizer_apply_chat_template(const Tokenizer *t, const char *user_text,
                                   uint32_t *out_ids, int max_ids);
#endif
