#!/bin/bash
# h100/run_round2.sh - round-2 unattended stock-vLLM run on ONE p5.48xlarge (8x H100 80GB, Deep Learning Base GPU AMI,
# Ubuntu 24.04). Same structure as run_all.sh (round 1): every step is logged, retried once, the run continues to the
# next step on failure, a second failure sends an SNS alert, results are uploaded after every step, then a final
# summary and `shutdown -h now`.
#
# Parts, in priority order (if time runs short the later parts are the ones that get cut):
#   d  chunked prefill fixed: round 1 started the server with the default --max-num-batched-tokens 8192 and the
#      references are at most 8,192 tokens long, so no sequence was longer than one chunk and chunked_prefill.py
#      exited after 1 s ("no reference sequence is longer than 8192 tokens"). Here the BF16 recipe is started with
#      --max-num-batched-tokens 2048 and 4096 (prompts of up to 8,192 tokens then span 4 and 2 chunks), plus a
#      2048 run without prefix caching, so no chunk can be served from the prefix cache.
#   a  per-instance measured: one server per GPU (8 replicas, model card 1x H100 recipe, ports 8500+gpu), the T16b
#      rows 8,192/512 and 1,024/1,024 at 32 in flight per replica, two repetitions with T16b's seeds (1234, 4321) and
#      prompt counts (bench/replica_bench.py), plus a one-GPU control in the same session.
#   b  FP8: the recipe plus `--quantization fp8` (vLLM online FP8, BF16 checkpoint) on GPU 0: Must matrix, GSM8K,
#      teacher-forced token match, tool calling.
#   c  MTP: the recipe plus `--speculative-config '{"method":"mtp","num_speculative_tokens":K}'` for K = 1, 2, 3
#      on GPU 0: c1 and c32 Must rows with the acceptance rate per row, GSM8K with its acceptance rate.
# FP8 and MTP are separate variants; nothing here changes the BF16 recipe used for the headline.
#
# Environment (all but BUCKET have defaults):
#   BUCKET            S3 bucket (required)         TOPIC_ARN   SNS topic for alerts and the summary
#   AWS_REGION        us-east-2                    WORK        work tree (default: parent of this directory)
#   DATA              /opt/dlami/nvme/nemotron (or /opt/nemotron/data)   models and caches
#   MAX_RUN_S         32400 (9 h)                  stop starting steps after this; the hard cap is HARD_CAP_MIN
#   HARD_CAP_MIN      600 (10 h)                   dead-man `shutdown -h +N`, set at start
#   REF_WAIT_MIN      15                           wait this long for the references (they exist from round 1)
#   REFS_NAME         references-dtnone            reference set under results/ and s3://$BUCKET/results/
#   NGPU              auto (nvidia-smi -L)         number of replicas for part a
#   RUN_ALL_NO_SHUTDOWN=1   do not shut down at the end (a wrapper does it; also for tests)
#   FORCE_PIP=1       skip docker and install vllm==0.27.1 with pip into a venv
#
# Security: every server binds 127.0.0.1 only (--host 127.0.0.1, docker --network host). Nothing is published.
# Results: $WORK/results/h100/round2/ -> s3://$BUCKET/results/h100/round2/ (round 1 in results/h100/ is not touched).

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
if [ -z "${DATA:-}" ]; then
  if [ -d /opt/dlami/nvme ] && [ -w /opt/dlami/nvme ]; then DATA=/opt/dlami/nvme/nemotron; else DATA=/opt/nemotron/data; fi
fi
MODEL_ROOT=$DATA/models
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
VENV=${VENV:-/opt/vllm-venv}
PORT=${PORT:-8000}                 # the single-GPU server (GPU 0)
REPLICA_PORT_BASE=${REPLICA_PORT_BASE:-8500}   # replica on GPU g listens on 8500+g (same as bench/opt/T11)
SERVED=nemotron
VARIANT="stock-vllm-$VLLM_VERSION-card-1xH100"
MAX_RUN_S=${MAX_RUN_S:-32400}
HARD_CAP_MIN=${HARD_CAP_MIN:-600}
REF_WAIT_MIN=${REF_WAIT_MIN:-15}
SEED_REP1=1234                     # T16b: repetition 1 = T11's seed
SEED_REP2=4321                     # T16b: repetition 2

RES=$WORK/results/h100/round2
LOGS=$RES/logs
STATE=$RES/state
SERVE_DIR=$RES/serve
R1=$WORK/results/h100-round1       # round-1 results pulled from S3 for the comparison tables (read only)
REFS_NAME=${REFS_NAME:-references-dtnone}
REFS=$WORK/results/$REFS_NAME
S3_RES=s3://$BUCKET/results/h100/round2/
mkdir -p "$LOGS" "$STATE" "$SERVE_DIR" "$MODEL_ROOT" "$DATA/cache/vllm" "$DATA/cache/flashinfer" "$DATA/cache/hf" "$LOGS/pids"

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
[ -f "$STATE/env" ] && . "$STATE/env"     # ENGINE, VLLM_BIN, SERVER_PHASE, NGPU, ... written by earlier steps
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

# The model card's 1x H100 recipe (README "1x H100 (vLLM)"), without the parser flags unless the phase says so.
# Phases:
#   base          the recipe as written
#   chunk2048 / chunk4096         base + --max-num-batched-tokens N  (part d)
#   chunk2048nopc                 chunk2048 with prefix caching off, so every chunk is computed (part d)
#   fp8 / fp8tools                base (+ parser flags) + --quantization fp8  (part b)
#   mtp1 / mtp2 / mtp3            base + --speculative-config '{"method":"mtp","num_speculative_tokens":K}'  (part c)
# Per-phase deviations written by the retry path: EXTRA_<phase> (extra flags, e.g. --max-model-len N) and
# NOPREFIX_<phase>=1 (MTP second attempt: prefix caching off, as the card's GB200 speculative recipe does).
serve_args() { # PHASE [PORT] -> one flag per line
  local ph=$1 port=${2:-$PORT} pc=--enable-prefix-caching npv=NOPREFIX_$1 ev=EXTRA_$1
  [ "${!npv:-0}" = 1 ] && pc=--no-enable-prefix-caching
  local parsers=(--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice)
  local a=(--model "$MODEL_DIR" --served-model-name "$SERVED" --host 127.0.0.1 --port "$port"
           --max-num-seqs 128 PC --async-scheduling
           --mamba-backend flashinfer --mamba-ssm-cache-dtype float16
           --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5)
  case "$ph" in
    base|replica) ;;
    chunk2048) a+=(--max-num-batched-tokens 2048) ;;
    chunk4096) a+=(--max-num-batched-tokens 4096) ;;
    chunk2048nopc) a+=(--max-num-batched-tokens 2048); pc=--no-enable-prefix-caching ;;
    fp8) a+=(--quantization fp8) ;;
    fp8tools) a+=(--quantization fp8 "${parsers[@]}") ;;
    mtp1|mtp2|mtp3) a+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${ph#mtp}}") ;;
    *) echo "unknown phase $ph" >&2; return 1 ;;
  esac
  local i
  for i in "${!a[@]}"; do [ "${a[$i]}" = PC ] && a[$i]=$pc; done
  # shellcheck disable=SC2206
  [ -n "${!ev:-}" ] && a+=(${!ev})
  printf '%s\n' "${a[@]}"
}

write_serve_file() { # PHASE -> $SERVE_DIR/serve-PHASE.sh (the exact command, for the writeup)
  local ph=$1 f="$SERVE_DIR/serve-$1.sh" args ev=EXTRA_$1 npv=NOPREFIX_$1
  mapfile -t args < <(serve_args "$ph")
  {
    echo "#!/bin/bash"
    echo "# Stock vLLM $VLLM_VERSION, 1x H100 80GB, model card recipe (phase: $ph). Written $(date -u +%FT%TZ)."
    echo "# Run through: docker run --gpus device=<gpu> --network host --ipc=host --entrypoint vllm $IMAGE serve ..."
    [ -n "${!ev:-}" ] && echo "# DEVIATION from the recipe (retry path): ${!ev}"
    [ "${!npv:-0}" = 1 ] && [ "$ph" != chunk2048nopc ] && echo "# DEVIATION from the recipe (retry path): prefix caching off"
    printf 'vllm serve'; printf ' %q' "${args[@]}"; echo
  } > "$f"
}

# ---- server control. NAME = main (GPU 0, $PORT) or r<gpu> (replica, port 8500+gpu).
cname() { echo "vllm-r2-$1"; }
pidfile() { echo "$STATE/server-$1.pid"; }

srv_alive() { # NAME
  if [ "${ENGINE:-}" = docker ]; then
    [ "$(docker inspect -f '{{.State.Running}}' "$(cname "$1")" 2>/dev/null)" = true ]
  else
    [ -f "$(pidfile "$1")" ] && kill -0 "$(cat "$(pidfile "$1")")" 2>/dev/null
  fi
}

srv_healthy() { curl -sf --max-time 5 "http://127.0.0.1:${1:-$PORT}/v1/models" >/dev/null 2>&1; }

srv_dump_logs() { # NAME LABEL -> $LOGS/server-LABEL.log
  local f="$LOGS/server-$2.log"
  if [ "${ENGINE:-}" = docker ]; then docker logs "$(cname "$1")" > "$f" 2>&1 || true
  else cp -f "$LOGS/server-live-$1.log" "$f" 2>/dev/null || true; fi
}

srv_stop_one() { # NAME LABEL; safe when nothing runs. Pip route: stops the process group this script started (setsid).
  local n=$1 label=${2:-stopped-$(date +%H%M%S)} pid i
  if [ "${ENGINE:-}" = docker ]; then
    if docker inspect "$(cname "$n")" >/dev/null 2>&1; then srv_dump_logs "$n" "$label"; docker rm -f "$(cname "$n")" >/dev/null 2>&1; fi
  elif [ -f "$(pidfile "$n")" ]; then
    pid=$(cat "$(pidfile "$n")")
    srv_dump_logs "$n" "$label"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for ((i = 0; i < 15; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; fi
    rm -f "$(pidfile "$n")"
  fi
}

gpus_free() { # "0 1 2" -> wait up to 2 min for GPU memory to come back
  local g used i
  for ((i = 0; i < 24; i++)); do
    local busy=0
    for g in $1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1 | tr -d ' ')
      if [ -n "$used" ] && [ "$used" -ge 2000 ]; then busy=1; fi
    done
    ((busy == 0)) && return 0
    sleep 5
  done
  log "WARNING: GPU(s) $1 still show memory in use after stop"
  return 0
}

srv_stop_main() { srv_stop_one main "${1:-stopped-$(date +%H%M%S)}"; setstate SERVER_PHASE none; SERVER_PHASE=none; gpus_free 0; }

srv_start() { # NAME GPU PORT PHASE
  local n=$1 gpu=$2 port=$3 ph=$4 args
  mapfile -t args < <(serve_args "$ph" "$port")
  log "starting server $n (GPU $gpu, port $port, phase $ph): vllm serve ${args[*]}"
  if [ "${ENGINE:-}" = docker ]; then
    docker rm -f "$(cname "$n")" >/dev/null 2>&1
    docker run -d --name "$(cname "$n")" --gpus "device=$gpu" --network host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v "$MODEL_ROOT:$MODEL_ROOT:ro" -v "$DATA/cache/vllm:/root/.cache/vllm" \
      -v "$DATA/cache/flashinfer:/root/.cache/flashinfer" -v "$DATA/cache/hf:/root/.cache/huggingface" \
      -e HF_HUB_OFFLINE=1 --entrypoint vllm "$IMAGE" serve "${args[@]}" >/dev/null
  else
    [ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
    CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="$DATA/cache/vllm" \
      setsid nohup "$VENV/bin/vllm" serve "${args[@]}" > "$LOGS/server-live-$n.log" 2>&1 &
    echo $! > "$(pidfile "$n")"
  fi
}

srv_wait() { # NAME PORT SECONDS -> 0 ready, 1 timeout, 2 server exited
  local end=$((SECONDS + $3))
  while ((SECONDS < end)); do
    srv_healthy "$2" && return 0
    srv_alive "$1" || return 2
    sleep 10
  done
  return 1
}

# ensure_server PHASE: leave a healthy single-GPU server (GPU 0) of that phase running, restarting it if needed.
# Returns 0 ok, 1 start failed, 3 gave up (this phase already failed to start twice; do not burn hours retrying).
ensure_server() {
  local ph=$1 n fails fvar=START_FAILS_$1 rc
  if srv_healthy "$PORT" && [ "${SERVER_PHASE:-none}" = "$ph" ]; then return 0; fi
  fails=${!fvar:-0}
  if ((fails >= 2)); then log "server phase $ph failed to start $fails times; giving up"; return 3; fi
  n=$(( ${SERVER_STARTS:-0} + 1 )); setstate SERVER_STARTS "$n"; SERVER_STARTS=$n
  if ((n > 20)); then log "too many server starts ($n)"; return 3; fi
  srv_stop_main "pre-start-$ph-$n"
  write_serve_file "$ph"
  srv_start main 0 "$PORT" "$ph" || { setstate "$fvar" $((fails + 1)); return 1; }
  srv_wait main "$PORT" 3000; rc=$?        # first start: weights 63 GB + FlashInfer JIT + CUDA graphs
  if ((rc != 0)); then
    log "server did not become ready (rc=$rc)"; srv_dump_logs main "failed-$ph-$n"
    tail -n 40 "$LOGS/server-failed-$ph-$n.log" 2>/dev/null
    setstate "$fvar" $((fails + 1))
    return 1
  fi
  setstate "$fvar" 0
  setstate SERVER_PHASE "$ph"; SERVER_PHASE=$ph
  log "server ready (phase $ph)"
  return 0
}

need_server() { # PHASE; propagates 3 (given up) so dependent steps are skipped, not retried
  ensure_server "$1"; local rc=$?
  ((rc == 0)) || echo "server ($1) not available (rc=$rc)"
  return $rc
}

server_maxlen() { # max_model_len the server reports (262144 if it cannot be read)
  local n
  n=$(curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len") or "")' 2>/dev/null)
  echo "${n:-262144}"
}

# ------------------------------------------------------------------ versions
imds() { # path under /latest/meta-data/
  local tok
  tok=$(curl -s --max-time 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 120')
  curl -s --max-time 3 -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"
}

record_versions() {
  local f=$RES/versions-h100-round2.txt
  {
    echo "# Versions for the H100 round-2 run, written $(date -u +%FT%TZ)"
    echo "platform: h100 (${NGPU:-?} GPUs visible; part a uses all of them, parts b-d use GPU 0)"
    echo "instance_type: $(imds instance-type)   az: $(imds placement/availability-zone)   ami: $(imds ami-id)"
    echo "os: $(. /etc/os-release; echo "$PRETTY_NAME")   kernel: $(uname -r)"
    echo "--- nvidia-smi"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv 2>&1
    nvidia-smi 2>&1 | sed -n '1,4p'
    nvidia-smi topo -m 2>&1 | head -12
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
    echo "bench/replica_bench.py md5: $(md5sum "$WORK/bench/replica_bench.py" 2>/dev/null | cut -d' ' -f1)"
    echo "model: $MODEL_ID  (weights from s3://<bucket>/models/$MODEL_NAME/, copied from Hugging Face)"
  } > "$f" 2>&1
  prun -m pip freeze > "$RES/pip-freeze-h100-round2.txt" 2>/dev/null || true
  # The exact flag spellings this vLLM accepts, for the two variants (evidence for the writeup).
  {
    echo "# vllm serve --help=all, lines around the flags used in round 2 ($(date -u +%FT%TZ))"
    if [ "${ENGINE:-}" = docker ]; then
      docker run --rm --entrypoint vllm "${CLIENT_IMAGE:-$IMAGE}" serve --help=all 2>&1
    else
      "$VENV/bin/vllm" serve --help=all 2>&1
    fi | grep -n -i -B1 -A6 -e '--quantization' -e '--speculative-config' -e '--max-num-batched-tokens' | head -150
  } > "$RES/vllm-serve-flags.txt" 2>&1 || true
}

# ------------------------------------------------------------------ steps
# Exit codes: 0 ok, 3 prerequisite missing (not retried, not alerted: the root cause alerted already), other = failure.

step_preflight() {
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; return 1; }
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  local n name; n=$(nvidia-smi -L | grep -c '^GPU')
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
  case "$name" in *H100*) ;; *) notify "[nemotron] H100 round 2: unexpected GPU" "GPU 0 is '$name', not an H100. Results are labelled h100 anyway; check before using them." ;; esac
  if [ -n "${NGPU_FORCE:-}" ]; then n=$NGPU_FORCE; fi
  setstate NGPU "$n"; NGPU=$n
  if ((n < 8)); then notify "[nemotron] H100 round 2: only $n GPUs" "Expected 8 H100 on a p5.48xlarge, nvidia-smi lists $n. Part a runs with $n replicas and says so."; fi
  df -h "$DATA" "$WORK" | tee -a "$RES/instance-meta.txt"
  free -g | tee -a "$RES/instance-meta.txt"
  command -v aws >/dev/null || return 1
  aws s3 ls "s3://$BUCKET/models/$MODEL_NAME/.complete" >/dev/null || { echo "cannot read the model in S3 (continuing, HF fallback)"; }
  { echo "instance_type: $(imds instance-type)"; echo "az: $(imds placement/availability-zone)"; echo "ami: $(imds ami-id)"; echo "gpus: $n"; } | tee -a "$RES/instance-meta.txt"
  aws configure set default.s3.max_concurrent_requests 64 || true
}

step_inputs() {
  mkdir -p "$WORK/bench"
  if [ ! -f "$WORK/bench/run_matrix.py" ] || [ ! -f "$WORK/bench/replica_bench.py" ]; then
    retry 3 aws s3 sync --only-show-errors "s3://$BUCKET/work/bench/" "$WORK/bench/" || return 1
  fi
  [ -f "$WORK/bench/run_matrix.py" ] && [ -f "$WORK/bench/accuracy/gsm8k.py" ] || { echo "bench/ incomplete"; return 1; }
  [ -f "$WORK/bench/replica_bench.py" ] || { echo "bench/replica_bench.py missing (part a needs it)"; return 1; }
  [ -f "$H100_DIR/spec_metrics.py" ] || { echo "h100/spec_metrics.py missing (part c needs it)"; return 1; }
  mkdir -p "$REFS" "$R1"
  aws s3 sync --only-show-errors "s3://$BUCKET/results/$REFS_NAME/" "$REFS/" || true
  log "reference files present: $(ls "$REFS"/*.jsonl 2>/dev/null | wc -l)"
  # Round-1 single-GPU rows and accuracy, for the comparison columns in the summary. Read only.
  aws s3 sync --only-show-errors "s3://$BUCKET/results/h100/baseline/" "$R1/baseline/" --exclude 'raw/*' || true
  aws s3 sync --only-show-errors "s3://$BUCKET/results/h100/accuracy/" "$R1/accuracy/" --exclude '*.details.jsonl' || true
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
      notify "[nemotron] H100 round 2: docker route failed, using pip" "docker pull or GPU passthrough failed for $IMAGE; installing vllm==$VLLM_VERSION with pip instead."
    fi
  fi
  if ((ok == 0)); then install_pip_engine || return 1; fi
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
  case "$v" in "$VLLM_VERSION"*) ;; *) notify "[nemotron] H100 round 2: vLLM version mismatch" "wanted $VLLM_VERSION, engine reports '$v'" ;; esac
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

# ---- single-GPU server steps
step_server() { # PHASE: start it (or confirm it) and run the sanity request
  local ph=$1
  [ -f "$MODEL_DIR/.verified" ] || { echo "model not ready"; return 3; }
  [ -n "${ENGINE:-}" ] || { echo "no engine"; return 3; }
  if [ "$STEP_ATTEMPT" -ge 2 ]; then
    # Second attempt after a failed start: apply the documented fallback and record it.
    local n; n=$(grep -ho 'maximum model length is [0-9]*' "$LOGS"/server-failed-"$ph"-*.log 2>/dev/null | tail -1 | grep -o '[0-9]*$')
    if [ -n "$n" ]; then setstate "EXTRA_$ph" "--max-model-len $n"; eval "EXTRA_$ph=\"--max-model-len $n\""; log "DEVIATION: --max-model-len $n"; fi
    case "$ph" in
      mtp*) setstate "NOPREFIX_$ph" 1; eval "NOPREFIX_$ph=1"; log "DEVIATION: $ph retried with prefix caching off" ;;
    esac
  fi
  ensure_server "$ph" || return $?
  local out="$RES/sanity-$ph.json"
  curl -sf --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one sentence.\"}],\"max_tokens\":200,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    > "$out" || { echo "sanity request failed"; return 1; }
  cat "$out"; echo
  python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; t=(m.get("content") or "")+(m.get("reasoning_content") or m.get("reasoning") or ""); sys.exit(0 if t.strip() else 1)' "$out" || { echo "empty sanity answer"; return 1; }
  srv_dump_logs main "ready-$ph"
  grep -ho 'max_num_batched_tokens[=:] *[0-9]*' "$LOGS/server-ready-$ph.log" 2>/dev/null | head -1 | tee "$STATE/max_num_batched_tokens-$ph.txt" || true
}
step_server_chunk2048() { step_server chunk2048; }
step_server_chunk4096() { step_server chunk4096; }
step_server_chunk2048nopc() { step_server chunk2048nopc; }
step_server_fp8() { step_server fp8; }
step_server_fp8tools() { step_server fp8tools; }
step_server_mtp1() { step_server mtp1; }
step_server_mtp2() { step_server mtp2; }
step_server_mtp3() { step_server mtp3; }
step_stop_server() { srv_stop_main "stop-$(date +%H%M%S)"; }

acc() { # PHASE OUTDIR SCRIPT OUTNAME extra args...   (writes OUTDIR/OUTNAME-h100.json)
  local ph=$1 outdir=$2 script=$3 name=$4; shift 4
  need_server "$ph" || return $?
  mkdir -p "$outdir"
  prun "bench/accuracy/$script.py" --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
    --variant "$VARIANT [round2 phase $ph]" --wait 60 --out "$outdir/$name-h100.json" "$@"
}

wait_for_refs() {
  local end=$((SECONDS + REF_WAIT_MIN * 60)) n
  while :; do
    aws s3 sync --only-show-errors "s3://$BUCKET/results/$REFS_NAME/" "$REFS/" >/dev/null 2>&1 || true
    n=$(ls "$REFS"/*.jsonl 2>/dev/null | wc -l)
    [ "$n" -ge 1 ] && return 0
    ((SECONDS >= end)) && return 1
    log "no references in S3 yet; waiting"
    sleep $(( end - SECONDS < 300 ? end - SECONDS + 1 : 300 ))
  done
}

need_refs() {
  wait_for_refs && return 0
  notify "[nemotron] H100 round 2: references missing" "No *.jsonl under s3://$BUCKET/results/$REFS_NAME/ after ${REF_WAIT_MIN} min. Steps that need them (chunked prefill, teacher-forced) are skipped."
  return 3
}

# ---- part d: chunked prefill
chunked() { # PHASE CHUNK
  local ph=$1 n=$2 seen
  need_server "$ph" || return $?
  seen=$(grep -ho 'max_num_batched_tokens=[0-9]*' "$LOGS/server-ready-$ph.log" 2>/dev/null | head -1 | grep -o '[0-9]*$')
  [ "$seen" = "$n" ] || { echo "server reports max_num_batched_tokens=${seen:-unknown}, wanted $n"; return 1; }
  need_refs || return 3
  acc "$ph" "$RES/chunked_prefill" chunked_prefill "chunked_prefill-$ph" --refs "$REFS" --max-num-batched-tokens "$n" --max-seq-len 8192
}
step_chunked_prefill_2048() { chunked chunk2048 2048; }
step_chunked_prefill_4096() { chunked chunk4096 4096; }
step_chunked_prefill_2048nopc() { chunked chunk2048nopc 2048; }

# ---- part a: 8 replicas, one per GPU
replica_names() { local g; for ((g = 0; g < ${NGPU:-8}; g++)); do printf 'r%s ' "$g"; done; }

# replicas_up: start every replica that is not healthy (GPU 0 first, so the shared compile and JIT caches are warm
# for the other seven), wait, and record the healthy GPU list in REPLICA_DEVS. Returns the number of healthy replicas
# through the variable HEALTHY_N; the caller decides what is enough.
replicas_up() {
  local n=${NGPU:-8} g port devs=() bad=() rc end
  if ! srv_healthy "$((REPLICA_PORT_BASE + 0))"; then
    srv_stop_one r0 "pre-start-r0"
    write_serve_file replica
    srv_start r0 0 "$REPLICA_PORT_BASE" replica
    srv_wait r0 "$REPLICA_PORT_BASE" 3000; rc=$?
    ((rc != 0)) && { log "replica r0 not ready (rc=$rc)"; srv_dump_logs r0 "failed-r0-$STEP_ATTEMPT"; }
  fi
  for ((g = 1; g < n; g++)); do
    port=$((REPLICA_PORT_BASE + g))
    if ! srv_healthy "$port"; then srv_stop_one "r$g" "pre-start-r$g"; srv_start "r$g" "$g" "$port" replica; fi
  done
  end=$((SECONDS + 3000))
  for ((g = 0; g < n; g++)); do
    port=$((REPLICA_PORT_BASE + g))
    while ! srv_healthy "$port" && srv_alive "r$g" && ((SECONDS < end)); do sleep 10; done
    if srv_healthy "$port"; then devs+=("$g"); else bad+=("$g"); srv_dump_logs "r$g" "failed-r$g-$STEP_ATTEMPT"; fi
  done
  HEALTHY_N=${#devs[@]}
  local list; list=$(IFS=,; echo "${devs[*]:-}")
  setstate REPLICA_DEVS "$list"; REPLICA_DEVS=$list
  log "replicas healthy: ${HEALTHY_N}/$n (GPUs: ${list:-none}); failed: ${bad[*]:-none}"
}

step_replicas_start() {
  [ -f "$MODEL_DIR/.verified" ] || { echo "model not ready"; return 3; }
  [ -n "${ENGINE:-}" ] || { echo "no engine"; return 3; }
  srv_stop_main "pre-replicas"
  replicas_up
  local n=${NGPU:-8}
  if ((HEALTHY_N == n)); then return 0; fi
  # Attempt 1 needs all of them. Attempt 2 (after one retry) continues with at least half and says so in the results.
  if [ "$STEP_ATTEMPT" -ge 2 ] && ((HEALTHY_N * 2 >= n)); then
    notify "[nemotron] H100 round 2: only $HEALTHY_N of $n replicas" "Part a continues with GPUs ${REPLICA_DEVS}. The result files list the devices used."
    return 0
  fi
  echo "only $HEALTHY_N of $n replicas are healthy"
  return 1
}

replicas_check() { # restart missing replicas; need at least half
  local n=${NGPU:-8}
  replicas_up
  ((HEALTHY_N * 2 >= n)) || { echo "only $HEALTHY_N of $n replicas healthy"; return 3; }
  return 0
}

row_done() { # JSON -> 0 if it exists with no failed requests and no failed clients
  python3 - "$1" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if d.get("failed") == 0 and d.get("clients_failed") == 0 and d.get("completed", 0) > 0 else 1)
PY
}

gpu_monitor_start() { # CSV
  setsid nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,clocks.sm,temperature.gpu,memory.used \
    --format=csv -l 20 > "$1" 2>&1 &
  echo $! > "$STATE/gpu-monitor.pid"
}
gpu_monitor_stop() {
  [ -f "$STATE/gpu-monitor.pid" ] || return 0
  local p; p=$(cat "$STATE/gpu-monitor.pid")
  kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
  rm -f "$STATE/gpu-monitor.pid"
}

replica_rep() { # REP SEED
  local rep=$1 seed=$2 row out="$RES/replicas/rep$1" rc=0
  replicas_check || return $?
  mkdir -p "$out"
  gpu_monitor_start "$out/gpu-monitor.csv"
  for row in agent_8k ref_1k_1k; do
    local nrep; nrep=$(echo "$REPLICA_DEVS" | tr ',' '\n' | grep -c .)
    if row_done "$out/${row}_${nrep}x32.json"; then log "$row rep$rep: already done"; continue; fi
    prun bench/replica_bench.py --row "$row" --devices "$REPLICA_DEVS" --conc-per 32 --seed "$seed" \
      --port-base "$REPLICA_PORT_BASE" --out-dir "$out" --tokenizer "$MODEL_DIR" --vllm-bin "${VLLM_BIN:-vllm}" \
      --note "H100 round 2: $nrep replicas (GPUs $REPLICA_DEVS), stock vLLM $VLLM_VERSION, model card 1x H100 recipe per GPU; repetition $rep (seed $seed); T16b method" || rc=1
    # replica_bench exits 0 even when every client failed; the result file is the judge.
    row_done "$out/${row}_${nrep}x32.json" || { echo "$row rep$rep: result has failed requests or failed clients"; rc=1; }
  done
  gpu_monitor_stop
  return $rc
}
step_replicas_rep1() { replica_rep 1 "$SEED_REP1"; }
step_replicas_rep2() { replica_rep 2 "$SEED_REP2"; }

step_replicas_ctrl1() { # the same two rows on GPU 0 alone, same session (scaling-efficiency control)
  local out="$RES/replicas/ctrl-1gpu" row rc=0
  srv_healthy "$REPLICA_PORT_BASE" || { echo "replica on GPU 0 is not up"; return 3; }
  mkdir -p "$out"
  for row in agent_8k ref_1k_1k; do
    row_done "$out/${row}_1x32.json" && continue
    prun bench/replica_bench.py --row "$row" --devices 0 --conc-per 32 --seed "$SEED_REP1" \
      --port-base "$REPLICA_PORT_BASE" --out-dir "$out" --tokenizer "$MODEL_DIR" --vllm-bin "${VLLM_BIN:-vllm}" \
      --note "H100 round 2: one-GPU control in the same session (replica on GPU 0 only, others idle), seed $SEED_REP1" || rc=1
    row_done "$out/${row}_1x32.json" || { echo "$row control: result has failed requests or failed clients"; rc=1; }
  done
  return $rc
}

step_replicas_stop() {
  local n
  for n in $(replica_names); do srv_stop_one "$n" "stop-$n-$(date +%H%M%S)"; done
  gpus_free "$(seq -s ' ' 0 $(( ${NGPU:-8} - 1 )))"
  setstate SERVER_PHASE none
}

# ---- part b: FP8
step_fp8_notes() { # what the flag quantizes, from the engine's own source and the server log
  local out="$RES/fp8/quantization-notes.txt"
  mkdir -p "$RES/fp8"
  {
    echo "# vLLM $VLLM_VERSION --quantization fp8 on the BF16 checkpoint ($(date -u +%FT%TZ))"
    echo "# server flags: $(grep -h '^vllm serve' "$SERVE_DIR/serve-fp8.sh" 2>/dev/null)"
    echo "## from the server log (loading and quantization lines)"
    grep -h -i -e 'quantization' -e 'fp8' -e 'Model loading took' -e 'Loading weights took' -e 'weight memory' \
      "$LOGS/server-ready-fp8.log" 2>/dev/null | cut -c1-400 | head -40
    echo "## engine source: which layers get which method (Fp8Config.get_quant_method)"
    prun - <<'PY' 2>&1
import inspect
try:
    from vllm.model_executor.layers.quantization import fp8
    print(inspect.getsource(fp8.Fp8Config.get_quant_method))
    from vllm.model_executor.layers.quantization.online import fp8 as o
    for cls in ("Fp8PerTensorOnlineLinearMethod", "Fp8PerTensorOnlineMoEMethod"):
        c = getattr(o, cls, None)
        print("###", cls, "-", (inspect.getdoc(c) or "").split("\n")[0] if c else "not found")
except Exception as e:
    print("could not read the source:", repr(e))
PY
  } > "$out" 2>&1
  [ -s "$out" ]
}

step_matrix_fp8() {
  need_server fp8 || return $?
  prun bench/run_matrix.py --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
    --max-model-len "$(server_maxlen)" --out-dir "$RES/fp8/matrix" --variant "$VARIANT + --quantization fp8" \
    --serve-cmd-file "$SERVE_DIR/serve-fp8.sh" \
    --note "stock vLLM $VLLM_VERSION, model card 1x H100 recipe + --quantization fp8 (online FP8 of the BF16 checkpoint), Must rows, temperature 1.0 top_p 0.95, --ignore-eos" \
    --wait 60 --vllm-bin "${VLLM_BIN:-vllm}"
}
step_gsm8k_fp8() { acc fp8 "$RES/fp8/accuracy" gsm8k gsm8k; }
step_teacher_forced_fp8() {
  need_refs || return 3
  acc fp8 "$RES/fp8/accuracy" teacher_forced teacher_forced --refs "$REFS" --mode auto --max-seq-len 8192
}
step_toolcall_fp8() { acc fp8tools "$RES/fp8/accuracy" toolcall toolcall; }

# ---- part c: MTP
mtp_matrix() { # K: c1 and c32 Must rows, one run_matrix call per row so the acceptance rate is per row
  local k=$1 ph=mtp$1 out="$RES/mtp/k$1" spec row c rc=0 acc="$RES/mtp/k$1/acceptance"
  need_server "$ph" || return $?
  mkdir -p "$acc"
  for spec in agent_8k:1 agent_8k:32 reasoning_1k_8k:1 reasoning_1k_8k:32 ref_1k_1k:1 ref_1k_1k:32 long_32k:1; do
    row=${spec%%:*}; c=${spec##*:}
    need_server "$ph" || return $?
    python3 "$H100_DIR/spec_metrics.py" snapshot --port "$PORT" --out "$acc/.before.json" || echo "WARNING: no spec-decode metrics before $row c$c"
    prun bench/run_matrix.py --platform h100 --port "$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" \
      --max-model-len "$(server_maxlen)" --out-dir "$out/matrix" --variant "$VARIANT + MTP k=$k" \
      --serve-cmd-file "$SERVE_DIR/serve-$ph.sh" --rows "$row" --conc "$c" \
      --note "stock vLLM $VLLM_VERSION, model card 1x H100 recipe + --speculative-config method=mtp k=$k, c1 and c32 Must rows, temperature 1.0 top_p 0.95, --ignore-eos. Acceptance counters cover the row's warmup requests too." \
      --wait 60 --vllm-bin "${VLLM_BIN:-vllm}" || rc=1
    python3 "$H100_DIR/spec_metrics.py" snapshot --port "$PORT" --out "$acc/.after.json" &&
      python3 "$H100_DIR/spec_metrics.py" delta "$acc/.before.json" "$acc/.after.json" --out "$acc/${row}_c${c}.json.new" --label "${row}_c${c}" &&
      mv -f "$acc/${row}_c${c}.json.new" "$acc/${row}_c${c}.json"
    rm -f "$acc"/.before.json "$acc"/.after.json "$acc"/*.json.new
  done
  return $rc
}
step_matrix_mtp1() { mtp_matrix 1; }
step_matrix_mtp2() { mtp_matrix 2; }
step_matrix_mtp3() { mtp_matrix 3; }

mtp_gsm8k() { # K
  local ph=mtp$1 acc="$RES/mtp/k$1/acceptance" rc
  need_server "$ph" || return $?
  mkdir -p "$acc"
  python3 "$H100_DIR/spec_metrics.py" snapshot --port "$PORT" --out "$acc/.before.json" || echo "WARNING: no spec-decode metrics before gsm8k"
  acc "$ph" "$RES/mtp/k$1/accuracy" gsm8k gsm8k; rc=$?
  python3 "$H100_DIR/spec_metrics.py" snapshot --port "$PORT" --out "$acc/.after.json" &&
    python3 "$H100_DIR/spec_metrics.py" delta "$acc/.before.json" "$acc/.after.json" --out "$acc/gsm8k.json" --label gsm8k
  rm -f "$acc"/.before.json "$acc"/.after.json
  return $rc
}
step_gsm8k_mtp1() { mtp_gsm8k 1; }
step_gsm8k_mtp2() { mtp_gsm8k 2; }
step_gsm8k_mtp3() { mtp_gsm8k 3; }

# ---- summary
step_summary() { summary_text; }
step_summary_text() { summary_text; }

summary_text() {
  python3 - "$RES" "$R1" <<'PY'
import csv, glob, json, os, sys
res, r1 = sys.argv[1], sys.argv[2]
out = []

def r1f(x, nd=1):
    try: return f"{float(x):.{nd}f}"
    except Exception: return "-"

steps = []
p = os.path.join(res, "state", "steps.tsv")
if os.path.exists(p):
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5: steps.append(f)
out.append("## Steps\n\n| Step | Status | Attempts | Seconds | Exit |\n|---|---|---|---|---|")
for f in steps: out.append("| " + " | ".join(f[:5]) + " |")

def load_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []
base = {(r["workload"], r["concurrency"]): r for r in load_csv(os.path.join(r1, "baseline", "matrix.csv"))}
base_acc = {}
for f in glob.glob(os.path.join(r1, "accuracy", "*-h100.json")):
    try: d = json.load(open(f)); base_acc[d.get("test")] = d
    except Exception: pass

def matrix_table(title, path, cmp_label="BF16 1 GPU (round 1)"):
    rows = load_csv(path)
    out.append(f"\n## {title}\n\n| Workload | Conc | Output tok/s | TTFT p50 ms | TPOT p50 ms | Status | {cmp_label} tok/s | Ratio |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = base.get((r["workload"], r["concurrency"]))
        bt = b.get("output_tok_s") if b else None
        try: ratio = f"{float(r['output_tok_s']) / float(bt):.2f}x"
        except Exception: ratio = "-"
        out.append(f"| {r['workload']} | {r['concurrency']} | {r1f(r.get('output_tok_s'))} | {r1f(r.get('ttft_p50_ms'))} | {r1f(r.get('tpot_p50_ms'))} | {r['status']} | {r1f(bt)} | {ratio} |")
    if not rows: out.append("| (none) | | | | | | | |")
    return rows

def acc_table(title, d, with_base=True):
    out.append(f"\n## {title}\n\n| Test | Result | Pass | BF16 round 1 |\n|---|---|---|---|")
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "*-h100.json"))):
        try: j = json.load(open(f))
        except Exception: continue
        if "test" not in j: continue
        n += 1
        b = base_acc.get(j.get("test")) if with_base else None
        out.append(f"| {os.path.basename(f)[:-10]} | {j.get('headline')} | {j.get('passed')} | {b.get('headline') if b else '-'} |")
    if not n: out.append("| (none) | | | |")
    return n

short = []
ok = sum(1 for f in steps if f[1] == "ok"); bad = [f[0] for f in steps if f[1] == "failed"]; skp = [f[0] for f in steps if f[1] == "skipped"]
short.append(f"steps ok {ok}/{len(steps)}; failed: {', '.join(bad) or 'none'}; skipped: {', '.join(skp) or 'none'}")

# part d
acc_table("Part d: chunked prefill (BF16 recipe, smaller --max-num-batched-tokens)", os.path.join(res, "chunked_prefill"), with_base=False)
for f in sorted(glob.glob(os.path.join(res, "chunked_prefill", "*-h100.json"))):
    try: j = json.load(open(f)); short.append(f"{os.path.basename(f)[:-10]}: {j.get('headline')} (passed={j.get('passed')})")
    except Exception: pass

# part a
out.append("\n## Part a: per-instance, one server per GPU, 32 in flight per replica\n")
out.append("| Run | Row | Replicas | In flight | Output tok/s | Sum of replica tok/s | TTFT p50 ms | TPOT p50 ms | Completed | Failed | GPUs |\n|---|---|---|---|---|---|---|---|---|---|---|")
rep_rows = []
for d in sorted(glob.glob(os.path.join(res, "replicas", "*"))):
    for f in sorted(glob.glob(os.path.join(d, "*x32.json"))):
        try: j = json.load(open(f))
        except Exception: continue
        rep_rows.append((os.path.basename(d), j))
        out.append(f"| {os.path.basename(d)} | {j['row']} | {j['replicas']} | {j['total_concurrency']} | {r1f(j.get('output_tok_s'))} | {r1f(j.get('sum_of_replica_output_tok_s'))} | {r1f(j.get('ttft_p50_ms'))} | {r1f(j.get('tpot_p50_ms'))} | {j['completed']} | {j['failed']} | {','.join(map(str, j['devices']))} |")
if not rep_rows: out.append("| (none) | | | | | | | | | | |")
for name, j in rep_rows:
    b = base.get((j["row"], "32"))
    one = None
    try: one = float(b["output_tok_s"]) if b else None
    except Exception: pass
    est = f", 1-GPU round 1 x {j['replicas']} = {one * j['replicas']:.0f}" if one else ""
    short.append(f"{name} {j['row']} x{j['replicas']}: {r1f(j.get('output_tok_s'))} tok/s{est}, TTFT p50 {r1f(j.get('ttft_p50_ms'))} ms, TPOT p50 {r1f(j.get('tpot_p50_ms'))} ms, failed {j['failed']}")

# part b
rows = matrix_table("Part b: FP8 (--quantization fp8), Must matrix, 1 GPU", os.path.join(res, "fp8", "matrix", "matrix.csv"))
for r in rows:
    if r["status"] == "ok": short.append(f"fp8 {r['workload']} c{r['concurrency']}: {r1f(r.get('output_tok_s'))} tok/s, TTFT {r1f(r.get('ttft_p50_ms'))} ms, TPOT {r1f(r.get('tpot_p50_ms'))} ms")
    else: short.append(f"fp8 {r['workload']} c{r['concurrency']}: {r['status']}")
acc_table("Part b: FP8 accuracy", os.path.join(res, "fp8", "accuracy"))
for f in sorted(glob.glob(os.path.join(res, "fp8", "accuracy", "*-h100.json"))):
    try: j = json.load(open(f)); short.append(f"fp8 {j.get('test')}: {j.get('headline')} (passed={j.get('passed')})")
    except Exception: pass

# part c
for k in (1, 2, 3):
    d = os.path.join(res, "mtp", f"k{k}")
    if not os.path.isdir(d): continue
    rows = matrix_table(f"Part c: MTP k={k}, c1 and c32 Must rows, 1 GPU", os.path.join(d, "matrix", "matrix.csv"), "BF16 no spec (round 1)")
    out.append(f"\n### MTP k={k} acceptance\n\n| Row | Acceptance rate | Mean accepted length | Per-position rate | Draft tokens |\n|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(d, "acceptance", "*.json"))):
        try: j = json.load(open(f))
        except Exception: continue
        pp = ", ".join(f"{float(v):.2f}" for v in (j.get("per_position_rate") or {}).values())
        out.append(f"| {os.path.basename(f)[:-5]} | {r1f(j.get('acceptance_rate'), 3)} | {r1f(j.get('mean_acceptance_len'), 2)} | {pp} | {int(j.get('draft_tokens', 0))} |")
        if os.path.basename(f) in ("agent_8k_c1.json", "gsm8k.json"):
            short.append(f"mtp k={k} {os.path.basename(f)[:-5]}: acceptance {r1f(j.get('acceptance_rate'), 3)}, mean length {r1f(j.get('mean_acceptance_len'), 2)}")
    acc_table(f"MTP k={k} accuracy", os.path.join(d, "accuracy"))
    for f in sorted(glob.glob(os.path.join(d, "accuracy", "*-h100.json"))):
        try: j = json.load(open(f)); short.append(f"mtp k={k} {j.get('test')}: {j.get('headline')} (passed={j.get('passed')})")
        except Exception: pass
    for r in rows:
        if r["status"] == "ok" and r["concurrency"] in ("1", "32") and r["workload"] in ("agent_8k", "ref_1k_1k"):
            short.append(f"mtp k={k} {r['workload']} c{r['concurrency']}: {r1f(r.get('output_tok_s'))} tok/s, TPOT {r1f(r.get('tpot_p50_ms'))} ms")

open(os.path.join(res, "RUN_SUMMARY.md"), "w").write("\n".join(out) + "\n")
open(os.path.join(res, "state", "sns-summary.txt"), "w").write("\n".join(short) + "\n")
print("\n".join(out))
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
exec > >(tee -a "$LOGS/run_round2.log") 2>&1

upload() { aws s3 sync --only-show-errors "$RES/" "$S3_RES" --exclude '*.tmp' --exclude 'state/env' --exclude 'logs/server-live-*' 2>&1 | tail -3 || true; }

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
    gpu_monitor_stop
    ((attempt == 1)) && { log "retrying $name once in 30 s"; sleep 30; }
  done
  printf '%s\tfailed\t2\t%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" "$rc" >> "$STATE/steps.tsv"
  notify "[nemotron] H100 round 2 step FAILED: $name" "Step $name failed twice (exit $rc). Last log lines:
$(tail -n 30 "$LOGS/step-$name.attempt2.log" 2>/dev/null)

Logs: $S3_RES""logs/"
  upload
  return 1
}

part_done() { # label: partial summary by SNS and a summary refresh in S3
  bash "$SELF" __step summary_text >/dev/null 2>&1
  notify "[nemotron] H100 round 2 progress: $1 finished" "$(cat "$STATE/sns-summary.txt" 2>/dev/null)"
  upload
}

FINALIZED=0
finalize() {
  ((FINALIZED)) && return; FINALIZED=1
  trap - EXIT
  log "finalizing"
  [ -f "$STATE/env" ] && . "$STATE/env"
  gpu_monitor_stop
  srv_stop_main "final" 2>/dev/null
  for n in $(replica_names); do srv_stop_one "$n" "final-$n" 2>/dev/null; done
  bash "$SELF" __step summary > "$LOGS/step-summary.final.log" 2>&1 || true
  upload
  local body; body="$(cat "$STATE/sns-summary.txt" 2>/dev/null)"
  notify "[nemotron] H100 round 2 finished" "${body:-no summary produced}

Wall time: $(( ($(date +%s) - START_EPOCH) / 60 )) min. Results: $S3_RES  (RUN_SUMMARY.md, versions-h100-round2.txt, chunked_prefill/, replicas/, fp8/, mtp/)"
  upload
  if [ -z "${RUN_ALL_NO_SHUTDOWN:-}" ]; then
    log "shutdown -h now"
    shutdown -h now
  fi
}
trap finalize EXIT
trap 'log "signal received"; exit 143' TERM INT

[ -z "${RUN_ALL_NO_SHUTDOWN:-}" ] && { shutdown -h +"$HARD_CAP_MIN" "nemotron round 2: $HARD_CAP_MIN min cap" >/dev/null 2>&1 || true; }
log "run_round2.sh start: bucket=$BUCKET work=$WORK image=$IMAGE"
: > "$STATE/steps.tsv"
rm -f "$STATE/env"; touch "$STATE/env"; setstate START_EPOCH "$START_EPOCH"

run_step preflight 600
[ -f "$STATE/env" ] && . "$STATE/env"
run_step inputs 900
run_step engine 3600
[ -f "$STATE/env" ] && . "$STATE/env"
run_step model 5400

# ---- part d: chunked prefill
run_step server_chunk2048 3900
run_step chunked_prefill_2048 5400
run_step stop_server 600
run_step server_chunk4096 3900
run_step chunked_prefill_4096 5400
run_step stop_server 600
run_step server_chunk2048nopc 3900
run_step chunked_prefill_2048nopc 5400
run_step stop_server 600
part_done "part d (chunked prefill)"

# ---- part a: 8 replicas
run_step replicas_start 4200
[ -f "$STATE/env" ] && . "$STATE/env"
run_step replicas_rep1 9000
run_step replicas_rep2 9000
run_step replicas_ctrl1 3600
run_step replicas_stop 900
part_done "part a (8 replicas)"

# ---- part b: FP8
run_step server_fp8 3900
run_step fp8_notes 600
run_step matrix_fp8 18000
run_step gsm8k_fp8 5400
run_step teacher_forced_fp8 $((REF_WAIT_MIN * 60 + 7200))
run_step stop_server 600
run_step server_fp8tools 3900
run_step toolcall_fp8 5400
run_step stop_server 600
part_done "part b (FP8)"

# ---- part c: MTP
for K in 1 2 3; do
  run_step "server_mtp$K" 3900
  run_step "matrix_mtp$K" 10800
  run_step "gsm8k_mtp$K" 5400
  run_step stop_server 600
  part_done "part c (MTP k=$K)"
done

run_step summary 600
exit 0
