#!/bin/bash
# Stock vLLM 0.27.1, 1x H100 80GB, model card recipe (phase: base). Written 2026-11-12T15:49:35Z.
# Run through: docker run --gpus device=0 --network host --ipc=host --entrypoint vllm vllm/vllm-openai:v0.27.1 serve ...
vllm serve --model /opt/dlami/nvme/nemotron/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --served-model-name nemotron --host 127.0.0.1 --port 8000 --max-num-seqs 128 --enable-prefix-caching --async-scheduling --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5
