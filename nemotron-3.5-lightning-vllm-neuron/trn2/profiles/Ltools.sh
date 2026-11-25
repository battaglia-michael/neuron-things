#!/bin/bash
# Profile L plus tool-call and reasoning parsers (tool-calling accuracy run). Same graphs as L.sh.
# Usage: Ltools.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
# Same KV cap / decode buckets as L.sh (see the T09 note there): baseline-v1 layout (NEMOTRON_H_MOE_WT=none)
# fits 0.22 and buckets [1, 4, 8, 16, 32]; the opt1 layout (up) needs 0.20 and [1, 8, 16, 32].
case "${NEMOTRON_H_MOE_WT:-up}" in
  none) cap=0.22; buckets="[1, 4, 8, 16, 32]" ;;
  *)    cap=0.20; buckets="[1, 8, 16, 32]" ;;
esac
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-$cap}
export MAX_MODEL_LEN=40960 MAX_NUM_SEQS=32 DECODE_BUCKETS=${DECODE_BUCKETS:-$buckets} DECODE_CTX_BUCKETS="[2048, 10240]"
exec ./serve_nemotron.sh --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
