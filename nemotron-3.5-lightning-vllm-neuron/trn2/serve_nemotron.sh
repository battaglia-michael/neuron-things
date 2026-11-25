#!/bin/bash
# Serve NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 on vLLM Neuron (branch nemotron-h,
# contrib/models/nemotron_h), TP=4 on one Trainium2 device, bound to 127.0.0.1.
#
# Env knobs (defaults = T04 first working config):
#   NEURON_VISIBLE_DEVICES  core ids (default 0-3 = device 0)
#   MAX_MODEL_LEN           default 40960 (32K workload + 512 out + jitter)
#   MAX_NUM_SEQS            default 8
#   CHUNK                   prefill chunk = --max-num-batched-tokens = KV segment size (default 2048)
#   PREFILL_BUCKETS         num_batched_tokens_buckets (default [CHUNK])
#   DECODE_BUCKETS          num_seqs_buckets (default [1, MAX_NUM_SEQS])
#   DECODE_CTX_BUCKETS      decode_context_length_buckets JSON list (default unset: decode at max_model_len)
#   GMU                     --gpu-memory-utilization (default: vLLM Neuron default)
#   PORT                    default 8100
#   SERVED_NAME             default nemotron (same as the H100 run)
#   PREFIX_CACHING          1 = --enable-prefix-caching (Mamba align mode, T11 stretch row); default 0
#   SAMPLING                "greedy" (on-device all_greedy) or "full" (default full: temperature/top_p honoured)
#   SRV_CPUS                server CPU affinity (taskset -c): empty = unpinned (default); "numa" = the CPU affinity
#                           neuron-ls reports for the device of the first core in NEURON_VISIBLE_DEVICES (device 0:
#                           48-95,144-191); or an explicit CPU list. No graph/NEFF change (T-numa-pin-server).
set -euo pipefail
MODEL=${MODEL:-${MODEL_DIR:?MODEL_DIR not set}}
source "$VLLM_VENV/bin/activate"
unset NEURON_RT_VISIBLE_CORES VLLM_NEURON_CPU_MODE
export NEURON_VISIBLE_DEVICES=${NEURON_VISIBLE_DEVICES:-0-3}
export NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native}
# The runtime's default 30 s execution timeout killed the engine (NRT_TIMEOUT, "missing
# collectives status") while the host ran 24 neuronx-cc jobs (load average 491).
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-300}

# vLLM's torch.compile package cache is keyed by the vLLM config only, not by the model source or
# the NEMOTRON_H_* flags, so a package traced from older code (or other flag values) is restored
# without retracing (T09c, T14). Unless VLLM_CACHE_ROOT is given, use a root keyed by the
# nemotron_h source and the NEMOTRON_H_* environment. The NEFF cache (content-addressed) stays shared.
if [ -z "${VLLM_CACHE_ROOT:-}" ]; then
  _nh=${VLLM_NEURON_DIR:?set VLLM_NEURON_DIR to your vllm-neuron clone}/contrib/models/nemotron_h/src
  _key=$( { cat "$_nh"/*.py "$_nh"/kernels/*.py; env | grep '^NEMOTRON_H_' | sort || true; } | sha1sum | cut -c1-12)
  export VLLM_CACHE_ROOT=$HOME/.cache/vllm-nemotron_h-$_key
fi
export TORCH_NEURONX_NEFF_CACHE_DIR=${TORCH_NEURONX_NEFF_CACHE_DIR:-$HOME/.cache/neuron_libtorch/neuron/native_neff_cache}
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"

MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
CHUNK=${CHUNK:-2048}
PREFILL_BUCKETS=${PREFILL_BUCKETS:-[$CHUNK]}
DECODE_BUCKETS=${DECODE_BUCKETS:-[1, $MAX_NUM_SEQS]}
SAMPLING=${SAMPLING:-full}

extra_cfg=""
[ -n "${DECODE_CTX_BUCKETS:-}" ] && extra_cfg="\"decode_context_length_buckets\": $DECODE_CTX_BUCKETS,"
if [ "$SAMPLING" = greedy ]; then
  sampling_cfg='"on_device_sampling_config": {"all_greedy": true}'
else
  sampling_cfg='"on_device_sampling_config": {"all_greedy": false}'
fi
pc_flag=--no-enable-prefix-caching; [ "${PREFIX_CACHING:-0}" = 1 ] && pc_flag=--enable-prefix-caching
gmu_args=(); [ -n "${GMU:-}" ] && gmu_args=(--gpu-memory-utilization "$GMU")

# EXTRA_ADDITIONAL_CONFIG: optional extra top-level --additional-config keys, e.g.
#   '"neuron_profiler": {"output_dir": "..."}' (T08 profiling). Empty = unchanged config.
extra_top=""
[ -n "${EXTRA_ADDITIONAL_CONFIG:-}" ] && extra_top=", $EXTRA_ADDITIONAL_CONFIG"
NEURON_CONFIG="{\"neuron_config\": {\"num_batched_tokens_buckets\": $PREFILL_BUCKETS, \"kv_segment_size_buckets\": [$CHUNK], \"num_seqs_buckets\": $DECODE_BUCKETS, $extra_cfg $sampling_cfg}$extra_top}"
echo "neuron_config: $NEURON_CONFIG"
pin=()
if [ -n "${SRV_CPUS:-}" ]; then
  if [ "$SRV_CPUS" = numa ]; then
    _core=${NEURON_VISIBLE_DEVICES%%[-,]*}
    SRV_CPUS=$(neuron-ls --json-output 2>/dev/null | python3 -c "import json,sys; c=int(sys.argv[1]); print(next(d['cpu_affinity'] for d in json.load(sys.stdin) if c in d['neuroncore_ids']))" "$_core" 2>/dev/null || true)
    [ -z "$SRV_CPUS" ] && echo "WARNING: SRV_CPUS=numa: no neuron-ls CPU affinity for core $_core; server runs unpinned"
  fi
  if [ -n "$SRV_CPUS" ]; then
    echo "server CPU affinity: taskset -c $SRV_CPUS"
    pin=(taskset -c "$SRV_CPUS")
  fi
fi
set -x
exec "${pin[@]}" vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-nemotron}" \
  --host 127.0.0.1 --port "${PORT:-8100}" \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$CHUNK" \
  "$pc_flag" \
  "${gmu_args[@]}" \
  --additional-config "$NEURON_CONFIG" \
  "$@"
