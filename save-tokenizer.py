"""
Run this ONCE to extract the tokenizer from a GGUF file and save it locally.
This avoids the slow "building merges on the fly" step on every startup.

Usage:
    python save_tokenizer.py <path to .gguf file>

It will create a 'tokenizer/' folder next to the GGUF file.
"""
import sys
import os
from transformers import AutoTokenizer

if len(sys.argv) < 2:
    print("Usage: python save_tokenizer.py <path to .gguf file>", file=sys.stderr)
    sys.exit(1)

gguf_path = sys.argv[1]
output_dir = os.path.join(os.path.dirname(gguf_path), "tokenizer_" + gguf_path.split("/")[-1].replace(".gguf", ""))

print(f"Loading tokenizer from GGUF: {gguf_path}")
print("(This will be slow one last time...)")

tokenizer = AutoTokenizer.from_pretrained(
    os.path.dirname(gguf_path),
    gguf_file=os.path.basename(gguf_path),
)

tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to: {output_dir}")
print("Now add 'tokenizerPath' to your config JSON pointing to this folder.")
