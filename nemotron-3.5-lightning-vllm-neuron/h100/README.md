# H100 scripts

Stock vLLM 0.27.1 (`vllm/vllm-openai:v0.27.1`) on a p5.48xlarge with NVIDIA's single-H100 recipe for this model
(`serve-base.sh`; `serve-tools.sh` adds the tool-call and reasoning parsers for the tool-calling test).

| File | What it does |
|---|---|
| `run_all.sh` | One GPU: the benchmark matrix, the long-context rows and the accuracy tests. |
| `run_round2.sh` | One p5.48xlarge: 8 servers (one per GPU) for the whole-server numbers, FP8 (`--quantization fp8`), MTP speculative decoding (K = 1, 2, 3), chunked prefill. |
| `user-data*.sh` | EC2 user data that runs the script above, uploads results after every step, and shuts the instance down. |
| `spec_metrics.py` | Reads MTP acceptance from the server's `/metrics`. |

The scripts run unattended from EC2 user data with an S3 bucket you pass in (`BUCKET`): they fetch `bench/` from
`s3://$BUCKET/work/bench/`, the model from `s3://$BUCKET/models/` (Hugging Face fallback) and the reference labels
from `s3://$BUCKET/results/references-dtnone/`, and write to `s3://$BUCKET/results/h100/`. Copy this folder's
`bench/` and your reference labels to those prefixes of a private bucket, or edit the paths. Every server binds
127.0.0.1, so the instance needs no inbound rules.
