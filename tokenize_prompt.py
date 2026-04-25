#!/usr/bin/env python3
"""
tokenize_prompt.py — Produce a binary file consumable by infer's
--prompt-tokens flag.

Format:  [uint32 count][uint32 token_id × count]   (little-endian)

Usage:
    python tokenize_prompt.py "Explain relativity" prompt.bin
"""
import sys, struct
from transformers import AutoTokenizer

def main():
    if len(sys.argv) != 3:
        print("usage: tokenize_prompt.py <text> <output.bin>")
        sys.exit(1)
    text, out_path = sys.argv[1], sys.argv[2]

    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1", trust_remote_code=True)

    # Step 1: render the chat template as a plain string
    messages = [{"role": "user", "content": text}]
    prompt_str = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)

    # Step 2: tokenize that string into a flat list of ints
    ids = tok.encode(prompt_str, add_special_tokens=False)

    # Defensive: ensure flat list of ints
    assert isinstance(ids, list) and all(isinstance(x, int) for x in ids), \
        f"expected list[int], got {type(ids)}: {ids[:3]}"

    print(f"Encoded {len(ids)} tokens")
    print(f"First 20: {ids[:20]}")
    print(f"Rendered prompt string:\n---\n{prompt_str}\n---")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))         # count header
        for i in ids:
            f.write(struct.pack("<I", int(i)))

    print(f"Wrote {out_path}: {4 + len(ids)*4} bytes")

if __name__ == "__main__":
    main()
