#!/bin/bash

# Runs on a RTX 6000 Blackwell with 96 GB VRAM, using 4-bit quantization with NVIDIA's FP4 format.

# Will not run on a H100 GPU, and the H100 is not optimized for FP4, while the FP8 version is too large for the H100

bash install-python-deps.sh
bash download-model.sh Firworks/Behemoth-X-123B-v2.1-nvfp4