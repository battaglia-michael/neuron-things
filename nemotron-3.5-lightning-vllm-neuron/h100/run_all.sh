#!/bin/bash
# h100/run_all.sh - unattended stock-vLLM run on ONE H100 (p5.4xlarge, Deep Learning Base GPU AMI, Ubuntu 24.04).
#
# Flow (every step is logged, retried once, and the run continues to the next step on failure;
# every final failure sends an SNS alert):
#   preflight -> inputs (bench/, A03 references from S3) -> engine (docker image vllm/vllm-openai:v0.27.1,
#   pip fallback) -> model (S3, then Hugging Face) -> server "base" (model card 1x H100 recipe)
#   -> matrix (Must rows) -> batch invariance -> batch probe of early divergences -> GSM8K -> needle -> matrix (Stretch rows)
#   -> teacher-forced -> chunked prefill -> server "tools" (recipe + parser flags) -> tool calling
#   -> summary -> upload to s3://$BUCKET/results/h100/ -> SNS summary -> shutdown -h now.
#
# Environment (all but BUCKET have defaults):
#   BUCKET            S3 bucket (required)         TOPIC_ARN   SNS topic for alerts and the summary
#   AWS_REGION        us-east-2                    WORK        work tree (default: parent of this directory)
#   DATA              /opt/dlami/nvme/nemotron (or /opt/nemotron/data)   models and caches
#   MAX_RUN_S         39600 (11 h)                 stop starting steps after this; the instance cap is 12 h
#   REF_WAIT_MIN      60                           wait this long for A03 references before teacher-forced
#   REFS_NAME         references-dtnone            reference set under results/ and s3://$BUCKET/results/:
#                                                  HF fp32 with the Mamba dt clamp off (T03b decision; vLLM GPU
#                                                  never clamps). "references" = stock HF (clamped), the A03 set.
#   RUN_ALL_NO_SHUTDOWN=1   do not shut down at the end (a wrapper does it; also for tests)
#   FORCE_PIP=1       skip docker and install vllm==0.27.1 with pip into a venv
#
# Security: the server binds 127.0.0.1 only (--host 127.0.0.1, docker --network host). Nothing is published.
# Results: $WORK/results/h100/{baseline,accuracy,logs,state,versions-h100.txt,RUN_SUMMARY.md}

set -uo pipefail

SELF=$(readlink -f "$0")
H100_DIR=$(dirname "$SELF")
WORK=${WORK:-$(dirname "$H100_DIR")}
BUCKET=${BUCKET:?BUCKET must be set}
TOPIC_ARN=${TOPIC_ARN:-}
export AWS_REGION=${AWS_REGION:-us-east-2}
export AWS_DEFAULT_REGION=$AWS_REGION

MODEL_ID=${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
MODEL_NAME=${MODEL_ID##*/}
VLLM_VERSION=${VLLM_VERSION:-0.27.1}
IMAGE=${IMAGE:-vllm/vllm-openai:v$VLLM_VERSION}
# Models and caches: the p5 instance-store NVMe (mounted by the DLAMI at /opt/dlami/nvme) if present, else the root volume.
if [ -z "${DATA:-}" ]; then
  if [ -d /opt/dlami/nvme ] && [ -w /opt/dlami/nvme ]; then DATA=/opt/dlami/nvme/nemotron; else DATA=/opt/nemotron/data; fi
fi
MODEL_ROOT=$DATA/models
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
VENV=${VENV:-/opt/vllm-venv}
PORT=${PORT:-8000}
SERVED=nemotron
CONTAINER=vllm-server
VARIANT="stock-vllm-$VLLM_VERSION-card-1xH100"
MAX_RUN_S=${MAX_RUN_S:-39600}
REF_WAIT_MIN=${REF_WAIT_MIN:-60}

RES=$WORK/results/h100
LOGS=$RES/logs
ACC=$RES/accuracy
BASE=$RES/baseline
STATE=$RES/state
REFS_NAME=${REFS_NAME:-references-dtnone}
REFS=$WORK/results/$REFS_NAME
S3_RES=s3://$BUCKET/results/h100/
mkdir -p "$LOGS" "$ACC" "$BASE" "$STATE" "$MODEL_ROOT" "$DATA/cache/vllm" "$DATA/cache/flashinfer" "$DATA/cache/hf"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

notify() { # subject, message. Never fails the caller.
  [ -n "$TOPIC_ARN" ] || return 0
  local subj msg
  subj=$(printf '%s' "$1" | tr -d '\r\n' | LC_ALL=C tr -c '[:print:]' '?' | cut -c1-99)
  msg=$(printf '%s' "${2:-no details}" | head -c 20000)
  aws sns publish --topic-arn "$TOPIC_ARN" --subject "$subj" --message "$msg" >/dev/null 2>&1 \
    || log "SNS publish failed: $subj"
  return 0
}

setstate() { printf '%s=%q\n' "$1" "$2" >> "$STATE/env"; }
[ -f "$STATE/env" ] && . "$STATE/env"     # ENGINE, VLLM_BIN, SERVER_PHASE, ... written by earlier steps
START_EPOCH=${START_EPOCH:-$(date +%s)}
if ! grep -q '^START_EPOCH=' "$STATE/env" 2>/dev/null; then setstate START_EPOCH "$START_EPOCH"; fi
STEP_ATTEMPT=${STEP_ATTEMPT:-1}

retry() { # retry N cmd... : N tries, 15 s apart
  local n=$1 i; shift
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    log "retry $i/$n failed: $*"; sleep 15
  done
  return 1
}

# ------------------------------------------------------------------ engine abstraction
# prun: run python (client scripts) where vllm 0.27.1 is installed; cwd = $WORK.
prun() {
  if [ "${ENGINE:-}" = docker ]; then
    docker run --rm -i --label nemotron-client=1 --network host --ipc=host \
      -v "$WORK:$WORK" -v "$MODEL_ROOT:$MODEL_ROOT" -v "$DATA/cache/hf:/root/.cache/huggingface" \
      -w "$WORK" -e PYTHONUNBUFFERED=1 -e BENCH_PLATFORM=h100 -e BENCH_PORT="$PORT" \
      --entrypoint python3 "${CLIENT_IMAGE:-$IMAGE}" "$@"
  else
    ( cd "$WORK" && PYTHONUNBUFFERED=1 BENCH_PLATFORM=h100 BENCH_PORT="$PORT" "$VENV/bin/python" "$@" )
  fi
}

# serve_args PHASE -> prints one flag per line. base = model card 1x H100 recipe,
# tools = the same plus the parser flags the card lists in its vLLM section.
serve_args() {
  local a=(--model "$MODEL_DIR" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT"
           --max-num-seqs 128 --enable-prefix-caching --async-scheduling
           --mamba-backend flashinfer --mamba-ssm-cache-dtype float16
           --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5)
  if [ "$1" = tools ]; then
    a+=(--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice)
  fi
  # SERVE_EXTRA is only set by the retry path (for example --max-model-len N) and is recorded.
  # shellcheck disable=SC2206
  [ -n "${SERVE_EXTRA:-}" ] && a+=($SERVE_EXTRA)
  printf '%s\n' "${a[@]}"
}

write_serve_file() { # PHASE -> $RES/serve-PHASE.sh (the exact command, for the writeup)
  local ph=$1 f="$RES/serve-$1.sh" args
  mapfile -t args < <(serve_args "$ph")
  {
    echo "#!/bin/bash"
    echo "# Stock vLLM $VLLM_VERSION, 1x H100 80GB, model card recipe (phase: $ph). Written $(date -u +%FT%TZ)."
    echo "# Run through: docker run --gpus device=0 --network host --ipc=host --entrypoint vllm $IMAGE serve ..."
    [ -n "${SERVE_EXTRA:-}" ] && echo "# DEVIATION from the recipe (retry path): $SERVE_EXTRA"
    printf 'vllm serve'; printf ' %q' "${args[@]}"; echo
  } > "$f"
}

srv_alive() {
  if [ "${ENGINE:-}" = docker ]; then
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ]
  else
    [ -f "$STATE/server.pid" ] && kill -0 "$(cat "$STATE/server.pid")" 2>/dev/null
  fi
}

srv_healthy() { curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

srv_dump_logs() { # label
  local f="$LOGS/server-$1.log"
  if [ "${ENGINE:-}" = docker ]; then docker logs "$CONTAINER" > "$f" 2>&1 || true
  else cp -f "$LOGS/server-current.log" "$f" 2>/dev/null || true; fi
}

srv_stop() { # label for the log dump; safe to call when nothing runs
  local label=${1:-stopped-$(date +%H%M%S)} i
  if [ "${ENGINE:-}" = docker ]; then
    if docker inspect "$CONTAINER" >/dev/null 2>&1; then srv_dump_logs "$label"; docker rm -f "$CONTAINER" >/dev/null 2>&1; fi
  elif [ -f "$STATE/server.pid" ]; then
    srv_dump_logs "$label"
    pkill -TERM -P "$(cat "$STATE/server.pid")" 2>/dev/null; kill -TERM "$(cat "$STATE/server.pid")" 2>/dev/null
    sleep 10; pkill -KILL -f "vllm serve" 2>/dev/null; rm -f "$STATE/server.pid"
  fi
  setstate SERVER_PHASE none
  for ((i = 0; i < 24; i++)); do   # wait up to 2 min for GPU memory to come back
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$used" ] || [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
  log "WARNING: GPU 0 still shows ${used} MiB used after stop"
  return 0
}

srv_start() { # PHASE
  local ph=$1 args
  write_serve_file "$ph"
  mapfile -t args < <(serve_args "$ph")
  log "starting server, phase $ph: vllm serve ${args[*]}"
  if [ "${ENGINE:-}" = docker ]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    docker run -d --name "$CONTAINER" --gpus device=0 --network host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v "$MODEL_ROOT:$MODEL_ROOT:ro" -v "$DATA/cache/vllm:/root/.cache/vllm" \
      -v "$DATA/cache/flashinfer:/root/.cache/flashinfer" -v "$DATA/cache/hf:/root/.cache/huggingface" \
      -e HF_HUB_OFFLINE=1 --entrypoint vllm "$IMAGE" serve "${args[@]}" >/dev/null
  else
    export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="$DATA/cache/vllm"
    [ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
    nohup "$VENV/bin/vllm" serve "${args[@]}" > "$LOGS/server-current.log" 2>&1 &
    echo $! > "$STATE/server.pid"
  fi
}

srv_wait() { # seconds -> 0 ready, 1 timeout, 2 server exited
  local end=$((SECONDS + $1))
  while ((SECONDS < end)); do
    srv_healthy && return 0
    srv_alive || return 2
    sleep 10
  done
  return 1
}

# ensure_server PHASE: leave a healthy server of that phase running, restarting it if needed.
# Returns 0 ok, 1 start failed, 3 gave up (this phase already failed to start twice; do not burn hours retrying).
ensure_server() {
  local ph=$1 n fails fvar=START_FAILS_$1 rc
  if srv_healthy && [ "${SERVER_PHASE:-none}" = "$ph" ]; then return 0; fi
  fails=${!fvar:-0}
  if ((fails >= 2)); then log "server phase $ph failed to start $fails times; giving up"; return 3; fi
  n=$(( ${SERVER_STARTS:-0} + 1 )); setstate SERVER_STARTS "$n"; SERVER_STARTS=$n
  if ((n > 8)); then log "too many server starts ($n)"; return 3; fi
  srv_stop "pre-start-$ph-$n"
  srv_start "$ph" || { setstate "$fvar" $((fails + 1)); return 1; }
  srv_wait 3000; rc=$?        # first start: weights 63 GB + FlashInfer JIT + CUDA graphs
  if ((rc != 0)); then
    log "server did not become ready (rc=$rc)"; srv_dump_logs "failed-$ph-$n"
    tail -n 40 "$LOGS/server-failed-$ph-$n.log" 2>/dev/null
    setstate "$fvar" $((fails + 1))
    return 1
  fi
  setstate "$fvar" 0
  setstate SERVER_PHASE "$ph"; SERVER_PHASE=$ph
  log "server ready (phase $ph)"
  return 0
}

server_maxlen() { # max_model_len the server reports (262144 if it cannot be read)
  local n
  n=$(curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len") or "")' 2>/dev/null)
  echo "${n:-262144}"
}

# ------------------------------------------------------------------ versions
record_versions() {
  local f=$RES/versions-h100.txt
  {
    echo "# Versions for the H100 run, written $(date -u +%FT%TZ)"
    echo "platform: h100 (1 GPU used, GPU 0)"
    echo "instance_type: $(imds instance-type)   az: $(imds placement/availability-zone)   ami: $(imds ami-id)"
    echo "os: $(. /etc/os-release; echo "$PRETTY_NAME")   kernel: $(uname -r)"
    echo "--- nvidia-smi"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv 2>&1
    nvidia-smi 2>&1 | sed -n '1,4p'
    echo "--- engine: ${ENGINE:-unknown}"
    if [ "${ENGINE:-}" = docker ]; then
      echo "docker: $(docker --version 2>&1)"
      echo "image: $IMAGE"
      echo "image_digest: $(docker inspect -f '{{index .RepoDigests 0}}' "$IMAGE" 2>&1)"
      echo "image_created: $(docker inspect -f '{{.Created}}' "$IMAGE" 2>&1)"
    else
      echo "venv: $VENV  pip install vllm==$VLLM_VERSION"
    fi
    echo "--- python packages (inside the engine)"
    prun - <<'PY' 2>&1
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ("vllm", "torch", "transformers", "flashinfer-python", "flashinfer", "mamba-ssm", "triton", "httpx", "xformers", "numpy"):
    try: print(p, md.version(p))
    except Exception: pass
try:
    import torch; print("torch.cuda", torch.version.cuda)
except Exception as e: print("torch import failed", e)
PY
    echo "--- harness"
    echo "bench/run_matrix.py md5: $(md5sum "$WORK/bench/run_matrix.py" 2>/dev/null | cut -d' ' -f1)"
    echo "model: $MODEL_ID  (weights from s3://<bucket>/models/$MODEL_NAME/, copied from Hugging Face)"
    [ -f "$MODEL_DIR/config.json" ] && echo "model max_position_embeddings: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_position_embeddings"))' "$MODEL_DIR/config.json")"
  } > "$f" 2>&1
  prun -m pip freeze > "$RES/pip-freeze-h100.txt" 2>/dev/null || true
}

imds() { # path under /latest/meta-data/
  local tok
  tok=$(curl -s --max-time 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 120')
  curl -s --max-time 3 -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"
}

# ------------------------------------------------------------------ steps
# Exit codes: 0 ok, 3 prerequisite missing (not retried, not alerted: the root cause alerted already), other = failure.

step_preflight() {
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; return 1; }
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
  case "$name" in *H100*) ;; *) notify "[nemotron] H100 run: unexpected GPU" "GPU 0 is '$name', not an H100. Results are labelled h100 anyway; check before using them." ;; esac
  [ "$(nvidia-smi -L | wc -l)" -gt 1 ] && log "NOTE: more than one GPU visible; only GPU 0 is used (docker --gpus device=0)."
  df -h "$DATA" "$WORK" | tee -a "$RES/instance-meta.txt"
  free -g | tee -a "$RES/instance-meta.txt"
  command -v aws >/dev/null || return 1
  aws s3 ls "s3://$BUCKET/models/$MODEL_NAME/.complete" >/dev/null || { echo "cannot read the model in S3 (continuing, HF fallback)"; }
  { echo "instance_type: $(imds instance-type)"; echo "az: $(imds placement/availability-zone)"; echo "ami: $(imds ami-id)"; } | tee -a "$RES/instance-meta.txt"
  aws configure set default.s3.max_concurrent_requests 64 || true
}

step_inputs() {
  mkdir -p "$WORK/bench"
  if [ ! -f "$WORK/bench/run_matrix.py" ]; then
    retry 3 aws s3 sync --only-show-errors "s3://$BUCKET/work/bench/" "$WORK/bench/" || return 1
  fi
  [ -f "$WORK/bench/run_matrix.py" ] && [ -f "$WORK/bench/accuracy/gsm8k.py" ] || { echo "bench/ incomplete"; return 1; }
  mkdir -p "$REFS"
  aws s3 sync --only-show-errors "s3://$BUCKET/results/$REFS_NAME/" "$REFS/" || true
  local n; n=$(ls "$REFS"/*.jsonl 2>/dev/null | wc -l)
  log "reference files present: $n"
  # Missing references are not fatal here: A03 may still be running. teacher_forced waits for them.
  return 0
}

install_pip_engine() {
  log "installing vllm==$VLLM_VERSION with pip into $VENV"
  apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || return 1
  "$VENV/bin/pip" install -q --upgrade pip &&
  "$VENV/bin/pip" install -q "vllm==$VLLM_VERSION" httpx pandas datasets || return 1
  setstate ENGINE pip; ENGINE=pip
  setstate VLLM_BIN "$VENV/bin/vllm"; VLLM_BIN=$VENV/bin/vllm
}

client_check() {
  prun -c 'import vllm, httpx, transformers; print("vllm", vllm.__version__)' || return 1
  if [ "${ENGINE:-}" = docker ]; then
    docker run --rm --network host --entrypoint vllm "${CLIENT_IMAGE:-$IMAGE}" bench serve --help >/dev/null 2>&1
  else
    "$VENV/bin/vllm" bench serve --help >/dev/null 2>&1
  fi
}

step_engine() {
  local ok=0
  if [ -z "${FORCE_PIP:-}" ] && command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    if retry 3 docker pull "$IMAGE" && docker run --rm --gpus device=0 --entrypoint nvidia-smi "$IMAGE" -L; then
      setstate ENGINE docker; ENGINE=docker
      setstate VLLM_BIN vllm; VLLM_BIN=vllm
      ok=1
    else
      log "docker route failed; falling back to pip"
      notify "[nemotron] H100: docker route failed, using pip" "docker pull or GPU passthrough failed for $IMAGE; installing vllm==$VLLM_VERSION with pip instead."
    fi
  fi
  if ((ok == 0)); then install_pip_engine || return 1; fi
  # Client self-check: the bench scripts need vllm, httpx, transformers; `vllm bench serve` needs its extras.
  if ! client_check; then
    log "client self-check failed; adding httpx pandas datasets"
    if [ "$ENGINE" = docker ]; then
      printf 'FROM %s\nRUN python3 -m pip install -q httpx pandas datasets || uv pip install --system -q httpx pandas datasets\n' "$IMAGE" |
        docker build -q -t nemotron-vllm-client - && { setstate CLIENT_IMAGE nemotron-vllm-client; CLIENT_IMAGE=nemotron-vllm-client; } || return 1
    else
      "$VENV/bin/pip" install -q httpx pandas datasets || return 1
    fi
    client_check || return 1
  fi
  local v; v=$(prun -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)
  case "$v" in "$VLLM_VERSION"*) ;; *) notify "[nemotron] H100: vLLM version mismatch" "wanted $VLLM_VERSION, engine reports '$v'" ;; esac
  record_versions
}

step_model() {
  if [ -f "$MODEL_DIR/.verified" ]; then log "model already verified"; return 0; fi
  if retry 2 aws s3 sync --only-show-errors "s3://$BUCKET/models/$MODEL_NAME/" "$MODEL_DIR/" --exclude '.cache/*'; then
    log "model copied from S3"
  else
    log "S3 copy failed; downloading from Hugging Face"
    prun -c "from huggingface_hub import snapshot_download as s; s('$MODEL_ID', local_dir='$MODEL_DIR')" || return 1
  fi
  python3 - "$MODEL_DIR" <<'PY' || { echo "model verification failed"; return 1; }
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
size = sum(os.path.getsize(os.path.join(d, f)) for f in files if f not in missing)
print(f"{len(files)} shards, {len(missing)} missing, {size/1e9:.1f} GB")
for need in ("config.json", "tokenizer.json", "chat_template.jinja"):
    if not os.path.exists(os.path.join(d, need)): missing.append(need)
sys.exit(1 if missing or size < 50e9 else 0)
PY
  touch "$MODEL_DIR/.verified"
}

step_server() { # PHASE
  local ph=$1
  [ -f "$MODEL_DIR/.verified" ] || { echo "model not ready"; return 3; }
  [ -n "${ENGINE:-}" ] || { echo "no engine"; return 3; }
  if [ "$STEP_ATTEMPT" -ge 2 ]; then
    # Second attempt: if vLLM said the context length does not fit, use the length it suggests and record the deviation.
    local n; n=$(grep -ho 'maximum model length is [0-9]*' "$LOGS"/server-failed-"$ph"-*.log 2>/dev/null | tail -1 | grep -o '[0-9]*$')
    if [ -n "$n" ]; then SERVE_EXTRA="--max-model-len $n"; setstate SERVE_EXTRA "--max-model-len $n"; log "DEVIATION: $SERVE_EXTRA"; fi
  fi
  ensure_server "$ph" || return $?
  local out="$RES/sanity-$ph.json"
  curl -sf --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one sentence.\"}],\"max_tokens\":200,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    > "$out" || { echo "sanity request failed"; return 1; }
  cat "$out"; echo
  python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; t=(m.get("content") or "")+(m.get("reasoning_content") or m.get("reasoning") or ""); sys.exit(0 if t.strip() else 1)' "$out" || { echo "empty sanity answer"; return 1; }
  # Server config line (chunk size etc.) for the record.
  srv_dump_logs "ready-$ph"
  grep -ho 'max_num_batched_tokens[=:] *[0-9]*' "$LOGS/server-ready-$ph.log" 2>/dev/null | head -1 | tee "$STATE/max_num_batched_tokens-$ph.txt" || true
}
step_server_base() { step_server base; }
step_server_tools() { step_server tools; }

need_server() { # PHASE; propagates 3 (given up) so dependent steps are skipped, not retried
  ensure_server "$1"; local rc=$?
  ((rc == 0)) || echo "server ($1) not available (rc=$rc)"
  return $rc
}

step_matrix_must() {
  need_server base || return $?
  prun bench/run_matrix.py --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
    --max-model-len "$(server_maxlen)" --out-dir "$BASE" --variant "$VARIANT" --serve-cmd-file "$RES/serve-base.sh" \
    --note "stock vLLM $VLLM_VERSION, model card 1x H100 recipe, temperature 1.0 top_p 0.95, --ignore-eos" \
    --wait 60 --vllm-bin "${VLLM_BIN:-vllm}"
}

step_matrix_stretch() {
  need_server base || return $?
  prun bench/run_matrix.py --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
    --max-model-len "$(server_maxlen)" --out-dir "$BASE" --variant "$VARIANT" --serve-cmd-file "$RES/serve-base.sh" \
    --note "stock vLLM $VLLM_VERSION, model card 1x H100 recipe, Stretch rows" \
    --rows long_128k,prefix_8k --wait 60 --vllm-bin "${VLLM_BIN:-vllm}"
}

acc() { # script name, extra args...  (writes $ACC/<name>-h100.json)
  local name=$1; shift
  need_server "${ACC_PHASE:-base}" || return $?
  prun "bench/accuracy/$name.py" --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
    --variant "$VARIANT" --wait 60 --out "$ACC/$name-h100.json" "$@"
}

step_batch_invariance() { acc batch_invariance; }

# Same batch-invariance rule as trn2 (T05/T05d): batch_invariance.py's own rule (<= 10% of prompts diverge
# before token 16) fails on bf16 logit ties alone. For every early-divergent prompt, batch_divergence_probe.py
# re-runs it solo (top-5 logprobs), in the mixed batch, and in a batch of 32 copies of itself; the summary step
# then calls combine_correctness.py --batch-probe, which judges by the leakage rule (tie at the divergence and
# the same divergence with identical batch-mates). No early divergences -> nothing to probe (skipped, rc 3).
early_divergent_ids() { # details.jsonl -> comma-separated ids diverging before params.min_prefix
  python3 - "$ACC/batch_invariance-h100.json" "$ACC/batch_invariance-h100.details.jsonl" <<'PY'
import json, sys
mp = json.load(open(sys.argv[1])).get("params", {}).get("min_prefix", 16)
ids = set()
for line in open(sys.argv[2]):
    r = json.loads(line)
    if r.get("first_divergence") is not None and r["first_divergence"] < mp:
        ids.add(r["id"])
print(",".join(sorted(ids)))
PY
}
step_batch_probe() {
  [ -f "$ACC/batch_invariance-h100.details.jsonl" ] || { echo "no batch_invariance details; nothing to probe"; return 3; }
  local ids; ids=$(early_divergent_ids) || { echo "could not read batch_invariance details"; return 1; }
  rm -f "$ACC/batch_divergence_probe-h100.json"
  [ -n "$ids" ] || { echo "no early-divergent prompts; batch probe not needed"; return 3; }
  log "batch probe on early-divergent prompts: $ids"
  need_server base || return $?
  prun bench/accuracy/batch_divergence_probe.py --port "$PORT" --model "$SERVED" --ids "$ids" \
    --out "$ACC/batch_divergence_probe-h100.json"
}
step_gsm8k() { acc gsm8k; }
step_needle() { need_server base || return $?; acc needle --max-model-len "$(server_maxlen)" --include-128k; }
step_toolcall() { ACC_PHASE=tools acc toolcall; }

wait_for_refs() {
  local end=$((SECONDS + REF_WAIT_MIN * 60)) n
  while :; do
    aws s3 sync --only-show-errors "s3://$BUCKET/results/$REFS_NAME/" "$REFS/" >/dev/null 2>&1 || true
    n=$(ls "$REFS"/*.jsonl 2>/dev/null | wc -l)
    [ "$n" -ge 1 ] && return 0
    ((SECONDS >= end)) && return 1
    log "no A03 references in S3 yet; waiting"
    sleep $(( end - SECONDS < 300 ? end - SECONDS + 1 : 300 ))
  done
}

step_teacher_forced() {
  if ! wait_for_refs; then
    notify "[nemotron] H100: teacher-forced skipped, no references" "No *.jsonl under s3://$BUCKET/results/$REFS_NAME/ after ${REF_WAIT_MIN} min (task A03 not done). Run bench/accuracy/teacher_forced.py against the H100 later, or re-run this step."
    return 3
  fi
  acc teacher_forced --refs "$REFS" --mode auto --max-seq-len 8192
}

step_chunked_prefill() {
  local f="$STATE/max_num_batched_tokens-base.txt" n
  n=$(grep -o '[0-9]*$' "$f" 2>/dev/null | head -1)
  [ -n "$n" ] || { echo "server log did not show max_num_batched_tokens; skipping chunked_prefill"; return 3; }
  ls "$REFS"/*.jsonl >/dev/null 2>&1 || { echo "no references"; return 3; }
  acc chunked_prefill --refs "$REFS" --max-num-batched-tokens "$n" --max-seq-len 8192
}

step_stop_server() { srv_stop "stop-$(date +%H%M%S)"; }

step_summary() {
  local probe=()
  [ -f "$ACC/batch_divergence_probe-h100.json" ] && probe=(--batch-probe "$ACC/batch_divergence_probe-h100.json")
  log "combine_correctness ${probe[*]:-(no batch probe)}"
  prun bench/accuracy/combine_correctness.py --platform h100 --dir "$ACC" "${probe[@]}" || true
  summary_text
}
step_summary_text() { summary_text; }

summary_text() {
  python3 - "$RES" <<'PY'
import csv, glob, json, os, sys
res = sys.argv[1]
out = []
steps = []
p = os.path.join(res, "state", "steps.tsv")
if os.path.exists(p):
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5: steps.append(f)
out.append("## Steps\n\n| Step | Status | Attempts | Seconds | Exit |\n|---|---|---|---|---|")
for f in steps: out.append("| " + " | ".join(f[:5]) + " |")
rows = []
mp = os.path.join(res, "baseline", "matrix.csv")
if os.path.exists(mp):
    rows = list(csv.DictReader(open(mp)))
out.append("\n## Benchmark rows (stock vLLM, 1x H100)\n\n| Workload | Conc | Output tok/s | TTFT p50 ms | TPOT p50 ms | Status |\n|---|---|---|---|---|---|")
def r1(x):
    try: return f"{float(x):.1f}"
    except Exception: return "-"
for r in rows:
    out.append(f"| {r['workload']} | {r['concurrency']} | {r1(r.get('output_tok_s'))} | {r1(r.get('ttft_p50_ms'))} | {r1(r.get('tpot_p50_ms'))} | {r['status']} |")
if not rows: out.append("| (none) | | | | | |")
out.append("\n## Accuracy\n\n| Test | Result | Pass |\n|---|---|---|")
accs = []
for f in sorted(glob.glob(os.path.join(res, "accuracy", "*-h100.json"))):
    try: d = json.load(open(f))
    except Exception: continue
    if "test" not in d: continue   # e.g. batch_divergence_probe-h100.json (raw probe rows)
    accs.append(d)
    out.append(f"| {d.get('test')} | {d.get('headline')} | {d.get('passed')} |")
if not accs: out.append("| (none) | | |")
text = "\n".join(out) + "\n"
open(os.path.join(res, "RUN_SUMMARY.md"), "w").write(text)
ok = sum(1 for f in steps if f[1] == "ok"); bad = [f[0] for f in steps if f[1] == "failed"]; skp = [f[0] for f in steps if f[1] == "skipped"]
short = [f"steps ok {ok}/{len(steps)}; failed: {', '.join(bad) or 'none'}; skipped: {', '.join(skp) or 'none'}"]
for r in rows:
    if r["status"] == "ok":
        short.append(f"{r['workload']} c{r['concurrency']}: {r1(r.get('output_tok_s'))} tok/s, TTFT {r1(r.get('ttft_p50_ms'))} ms, TPOT {r1(r.get('tpot_p50_ms'))} ms")
    else:
        short.append(f"{r['workload']} c{r['concurrency']}: {r['status']}")
for d in accs: short.append(f"{d.get('test')}: {d.get('headline')} (passed={d.get('passed')})")
open(os.path.join(res, "state", "sns-summary.txt"), "w").write("\n".join(short) + "\n")
print(text)
PY
}

# ------------------------------------------------------------------ step runner (child mode)
if [ "${1:-}" = __step ]; then
  fn="step_$2"
  declare -F "$fn" >/dev/null || { echo "unknown step $2"; exit 2; }
  "$fn"
  exit $?
fi

# ------------------------------------------------------------------ main
exec > >(tee -a "$LOGS/run_all.log") 2>&1

upload() { aws s3 sync --only-show-errors "$RES/" "$S3_RES" --exclude '*.tmp' --exclude 'state/env' 2>&1 | tail -3 || true; }

run_step() { # name timeout_s
  local name=$1 tmo=$2 attempt rc t0 left
  left=$(( MAX_RUN_S - ($(date +%s) - START_EPOCH) ))
  if ((left < 1200)); then
    log "STEP $name skipped: past the run deadline"
    printf '%s\tskipped\t0\t0\tdeadline\n' "$name" >> "$STATE/steps.tsv"; return 3
  fi
  ((tmo > left - 600)) && tmo=$((left - 600))
  for attempt in 1 2; do
    t0=$(date +%s)
    log "STEP $name: attempt $attempt (timeout ${tmo}s)"
    # shellcheck disable=SC1090
    [ -f "$STATE/env" ] && . "$STATE/env"
    STEP_ATTEMPT=$attempt timeout --kill-after=60 "$tmo" bash "$SELF" __step "$name" > "$LOGS/step-$name.attempt$attempt.log" 2>&1
    rc=$?
    log "STEP $name: attempt $attempt exit $rc after $(( $(date +%s) - t0 )) s"
    if ((rc == 0)); then
      printf '%s\tok\t%s\t%s\t0\n' "$name" "$attempt" "$(( $(date +%s) - t0 ))" >> "$STATE/steps.tsv"; upload; return 0
    fi
    if ((rc == 3)); then
      tail -n 5 "$LOGS/step-$name.attempt$attempt.log"
      printf '%s\tskipped\t%s\t%s\t3\n' "$name" "$attempt" "$(( $(date +%s) - t0 ))" >> "$STATE/steps.tsv"; upload; return 3
    fi
    tail -n 15 "$LOGS/step-$name.attempt$attempt.log"
    if command -v docker >/dev/null; then docker ps -q --filter label=nemotron-client=1 | xargs -r docker rm -f >/dev/null 2>&1; fi
    ((attempt == 1)) && { log "retrying $name once in 30 s"; sleep 30; }
  done
  printf '%s\tfailed\t2\t%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" "$rc" >> "$STATE/steps.tsv"
  notify "[nemotron] H100 step FAILED: $name" "Step $name failed twice (exit $rc). Last log lines:
$(tail -n 30 "$LOGS/step-$name.attempt2.log" 2>/dev/null)

Logs: $S3_RES""logs/"
  upload
  return 1
}

FINALIZED=0
finalize() {
  ((FINALIZED)) && return; FINALIZED=1
  trap - EXIT
  log "finalizing"
  [ -f "$STATE/env" ] && . "$STATE/env"
  srv_stop "final" 2>/dev/null
  bash "$SELF" __step summary > "$LOGS/step-summary.final.log" 2>&1 || true
  upload
  local body; body="$(cat "$STATE/sns-summary.txt" 2>/dev/null)"
  notify "[nemotron] H100 run finished" "${body:-no summary produced}

Wall time: $(( ($(date +%s) - START_EPOCH) / 60 )) min. Results: $S3_RES  (RUN_SUMMARY.md, versions-h100.txt, baseline/matrix.csv, accuracy/)"
  upload
  if [ -z "${RUN_ALL_NO_SHUTDOWN:-}" ]; then
    log "shutdown -h now"
    shutdown -h now
  fi
}
trap finalize EXIT
trap 'log "signal received"; exit 143' TERM INT

[ -z "${RUN_ALL_NO_SHUTDOWN:-}" ] && { shutdown -h +720 "nemotron 12 h cap" >/dev/null 2>&1 || true; }
log "run_all.sh start: bucket=$BUCKET work=$WORK image=$IMAGE"
: > "$STATE/steps.tsv"
rm -f "$STATE/env"; touch "$STATE/env"; setstate START_EPOCH "$START_EPOCH"

run_step preflight 600
run_step inputs 900
run_step engine 3600
[ -f "$STATE/env" ] && . "$STATE/env"
run_step model 5400
run_step server_base 3900

# Dependent steps call ensure_server themselves and restart the server if it died.
run_step matrix_must 18000
bash "$SELF" __step summary_text >/dev/null 2>&1
notify "[nemotron] H100 progress: Must matrix step finished" "$(cat "$STATE/sns-summary.txt" 2>/dev/null)"
run_step batch_invariance 5400
run_step batch_probe 3600
run_step gsm8k 5400
run_step needle 7200
run_step matrix_stretch 7200
run_step teacher_forced $((REF_WAIT_MIN * 60 + 7200))
run_step chunked_prefill 5400
run_step stop_server 600
run_step server_tools 3900
run_step toolcall 5400
run_step stop_server 600
run_step summary 600
exit 0
