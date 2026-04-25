#!/usr/bin/env python3
"""Export DeepSeek-R1 tokenizer to binary format consumable by tokenizer.c.

Source order:
  1. $MODEL_DIR/tokenizer.json (if present)
  2. HuggingFace hub: deepseek-ai/DeepSeek-R1 (cached by transformers)
"""
import json, struct, sys, os

MODEL = os.environ.get(
    "MODEL_DIR",
    os.path.expanduser("~/models/deepseek-r1-q4km/DeepSeek-R1-Q4_K_M"))

def load_tokenizer_json():
    local = os.path.join(MODEL, "tokenizer.json")
    if os.path.exists(local):
        print(f"[src] {local}")
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fall back to HF — uses transformers' cache, no extra download
    # if you've already run tokenize_prompt.py.
    print("[src] tokenizer.json not in model dir; fetching via transformers")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    # tok.vocab_file points at tokenizer.json (or its dir) for fast tokenizers.
    src_dir = os.path.dirname(tok.vocab_file) if getattr(tok, "vocab_file", None) else None
    cand = None
    if src_dir:
        c = os.path.join(src_dir, "tokenizer.json")
        if os.path.exists(c): cand = c
    if cand is None:
        # Save it out, then read it back.
        import tempfile
        d = tempfile.mkdtemp(prefix="dsr1tok_")
        tok.save_pretrained(d)
        cand = os.path.join(d, "tokenizer.json")
    print(f"[src] {cand}")
    with open(cand, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    t = load_tokenizer_json()

    model = t["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = t.get("added_tokens", [])

    # tokenizer.json sometimes stores merges as ["a b", ...] (string),
    # sometimes as [["a","b"], ...] (pair). Normalize.
    norm_merges = []
    for m in merges:
        if isinstance(m, str):
            sp = m.split(" ", 1)
            if len(sp) != 2:
                continue
            norm_merges.append((sp[0], sp[1]))
        else:
            norm_merges.append((m[0], m[1]))

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    out_path = "tokenizer.bin"

    with open(out_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(sorted_vocab)))
        f.write(struct.pack("<I", len(norm_merges)))
        f.write(struct.pack("<I", len(added)))

        for token_str, token_id in sorted_vocab:
            b = token_str.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        for a, b in norm_merges:
            ab = a.encode("utf-8")
            bb = b.encode("utf-8")
            f.write(struct.pack("<H", len(ab))); f.write(ab)
            f.write(struct.pack("<H", len(bb))); f.write(bb)

        for tok in added:
            b = tok["content"].encode("utf-8")
            f.write(struct.pack("<I", tok["id"]))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    sz = os.path.getsize(out_path)
    print(f"[ok] {out_path}: {len(sorted_vocab)} vocab, "
          f"{len(norm_merges)} merges, {len(added)} added "
          f"({sz/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
