#!/bin/bash

# Runs on a H200 GPU with 140 GB VRAM, using 8-bit quantization with NVIDIA's FP8 format.

# Will not run on a H100 GPU nor a RTX 6000 Blackwell, as the model is too large for the H100 and the FP8 format is not natively supported on the RTX 6000 Blackwell

bash install-python-deps.sh
./pyenv/bin/python download-model.py sh0ck0r/Behemoth-X-123B-v2.1-FP8-Dynamic