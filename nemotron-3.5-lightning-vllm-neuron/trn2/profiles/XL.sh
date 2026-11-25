#!/bin/bash
# Profile XL (extra long): 131,072/512 at c=1 (T11 stretch row). One sequence, one decode graph.
# Usage: XL.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-0.25}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-133120} MAX_NUM_SEQS=1 DECODE_BUCKETS="[1]"
exec ./serve_nemotron.sh
