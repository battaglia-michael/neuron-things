#!/bin/bash
# Profile P (prefix caching): profile S graphs with --enable-prefix-caching (Mamba align mode),
# for the shared-prefix 8,192/512 row at c=32 (T11 stretch row). 32 sequences, buckets 1/8/32.
# Usage: P.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-0.25}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8960} MAX_NUM_SEQS=${MAX_NUM_SEQS:-32} DECODE_BUCKETS="[1, 8, 32]"
# T11f: CHUNK=4096 + PREFILL_BUCKETS [2048, 4096] (as S/L) passes the gates but is 1.8% slower on prefix_8k c32
# (results/trn2/stretch/prefix-chunk4096/), so P keeps serve_nemotron.sh's CHUNK 2048.
export PREFIX_CACHING=1
exec ./serve_nemotron.sh
