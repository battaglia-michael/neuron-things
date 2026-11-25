#!/bin/bash
# Profile S (short): 8192/512 at c=1,32,128 and 1024/1024 at c=1,32.
# KV must hold 128 x max_model_len: the Neuron scheduler admits at most
# kv_tokens // max_model_len requests (see notes/troubleshooting.md).
# Usage: S.sh [cores, default 0-3] [server port, default 8100]
cd "$(dirname "$0")/.."   # trn2/
export NEURON_VISIBLE_DEVICES=${1:-0-3} PORT=${2:-8100}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.25
# No decode_context_length_buckets: with the 2048 context bucket the 9 graphs do not fit
# in HBM at load (Allocation Failure; DMA spill rings of the batch-128 graphs ~0.6 GB each).
export MAX_MODEL_LEN=8960 MAX_NUM_SEQS=128 DECODE_BUCKETS="[1, 8, 32, 128]"
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
