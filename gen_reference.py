#!/usr/bin/env python3
"""
Usage: gen_reference.py <tensor_name> [n_elements]
Prints the first n (default 16) dequantized fp32 values of <tensor_name>,
one per line, format %.6f — matching quant_test.c's stdout exactly.
"""
import sys, os
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

PATH = os.path.expanduser(
    "~/models/deepseek-r1-q4km/DeepSeek-R1-Q4_K_M/"
    "DeepSeek-R1-Q4_K_M-00001-of-00011.gguf"
)

def main():
    if len(sys.argv) < 2:
        print("usage: gen_reference.py <tensor_name> [n]", file=sys.stderr)
        sys.exit(1)
    name   = sys.argv[1]
    nprint = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    r = GGUFReader(PATH)
    for t in r.tensors:
        if t.name != name:
            continue

        qt = t.tensor_type
        if qt in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                  GGMLQuantizationType.BF16):
            arr = np.array(t.data, copy=False).astype(np.float32).reshape(-1)
        else:
            # Raw quant bytes -> fp32 via the reference dequantizer
            from gguf.quants import dequantize
            raw = np.array(t.data, copy=False)       # uint8 view
            arr = dequantize(raw, qt).astype(np.float32).reshape(-1)

        print(f"tensor {name}  n_elem={arr.size}  type={qt.name}",
              file=sys.stderr)
        for v in arr[:nprint]:
            print(f"{v:.6f}")
        return

    print(f"tensor not found: {name}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
