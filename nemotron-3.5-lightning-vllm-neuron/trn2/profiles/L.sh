#!/bin/bash
# Profile L (long): 32768/512 at c=1,16 and 1024/8192 at c=1,32.
# Usage: L.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
# KV cap: 0.22 for baseline-v1 (36.6 x 40960-token requests). With NEMOTRON_H_MOE_WT=up (T09 opt1)
# the 16 L graphs need ~1 MB more HBM than is left at 0.22 (load fails: "Failed to allocate
# 1001.000KB ... dma rings spill"). At 0.20 (4.42 GiB KV, 33 requests; the rows need 32) 16 of 17
# graphs load and the last one is ~100 MB short ("Failed to allocate 52.524MB ... model code").
# Lower caps admit < 32 requests, so the opt1 L profile also drops decode bucket 4 (unused by the
# c1/c16/c32 rows; 2-4 running requests pad to bucket 8). Context buckets and max-model-len unchanged.
case "${NEMOTRON_H_MOE_WT:-up}" in  # same default as contrib/models/nemotron_h/src/moe.py
  none) cap=0.22; buckets="[1, 4, 8, 16, 32]" ;;
  *)    cap=0.20; buckets="[1, 8, 16, 32]" ;;
esac
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-$cap}
export MAX_MODEL_LEN=40960 MAX_NUM_SEQS=32 DECODE_BUCKETS=${DECODE_BUCKETS:-$buckets} DECODE_CTX_BUCKETS="[2048, 10240]"
# T-chunk4096 (optimization step 15): 4,096-token prefill chunks (--max-num-batched-tokens 4096, kv segment 4096) plus a
# 2,048 bucket so a lone prompt of <= 2,048 tokens does not pad to 4,096 (each prefill step holds one sequence). Needs
# nemotron_h's default NEMOTRON_H_MAMBA_PREFILL_SUBCHUNK=1024 (one-pass Mamba prefill at 4,096 tokens is 11x slower).
# CHUNK=2048 PREFILL_BUCKETS="[2048]" = the configuration up to step 14.
export CHUNK=${CHUNK:-4096}
[ "$CHUNK" = 4096 ] && export PREFILL_BUCKETS=${PREFILL_BUCKETS:-"[2048, 4096]"}
# T-numa-pin-server: run the server on the CPUs of the device's NUMA node (neuron-ls CPU affinity; device 0: 48-95,144-191).
# c=1 TPOT -1..-2.5%, c>=16 rows within 0.1% (results/trn2/opt-numa-pin/). No graph or NEFF change. SRV_CPUS= (empty) = unpinned.
export SRV_CPUS=${SRV_CPUS-numa}
# R2-10b found vllm-neuron's shm_broadcast reader patch (0.2 ms spin + zmq-notify block) delayed the EngineCore wake-up by
# 0.55 ms per step, and =0 (upstream 1 s busy-yield) gave c=1 +2.4..+6.7% (results/trn2/r2-host/). After the host prep cache
# (R2-G, 66461c8) the loop has ~4.3 ms of slack: R2-10e (results/trn2/r2-host/spin-quiet/) measured =0 and =1 equal on a quiet
# device 0 (c=1 Must rows within 0.1% TPOT) and R2-10d the same under 16-server load, while =0 costs ~3.4 busy vCPUs per
# server. So the default is back to =1 (patch on). Runtime only, no graph/NEFF change. VLLM_NEURON_SHM_READER_SPIN_PATCH=0 = R2-F.
export VLLM_NEURON_SHM_READER_SPIN_PATCH=${VLLM_NEURON_SHM_READER_SPIN_PATCH-1}
exec ./serve_nemotron.sh
