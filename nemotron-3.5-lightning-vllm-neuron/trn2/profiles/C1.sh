#!/bin/bash
# Profile C1 (T09b, latency deployment): one request at a time, up to 40,960 tokens (all c=1 Must rows).
# Only the batch-1 decode graphs (same context buckets as L) and the 2,048-token prefill graph.
# With max-num-seqs 1 the KV cache needs ~0.14 GiB, so the cap drops to 0.01 (~0.22 GiB). The HBM
# this frees holds the second copy of the routed up weights that NEMOTRON_H_MOE_TKG=1 (default here)
# needs (+6.84 GiB per rank, [E, H, 1, I] layout for nkilib moe_tkg; weights 14.74 -> 21.58 GiB).
# At cap 0.02 the graphs do not load with the copy ("Allocation Failure", T09b).
# T09b, device 0: c1 TPOT 6.6-7.1 ms (NEMOTRON_H_MOE_TKG=0 in this profile: 9.0-9.4 ms).
# Usage: C1.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-0.01}
# vllm-neuron refuses a KV budget under 1 GiB by default; one 40,960-token request needs ~0.14 GiB.
export VLLM_NEURON_MIN_KV_BUDGET_GIB=${VLLM_NEURON_MIN_KV_BUDGET_GIB:-0.15}
export NEMOTRON_H_MOE_TKG=${NEMOTRON_H_MOE_TKG:-1}
export MAX_MODEL_LEN=40960 MAX_NUM_SEQS=1 DECODE_BUCKETS="[1]" DECODE_CTX_BUCKETS="[2048, 10240]"
exec bash ./serve_nemotron.sh
