#!/bin/bash
# h100/user-data-round2.sh - EC2 user data for the round-2 run (p5.48xlarge, 8x H100, Deep Learning Base GPU AMI,
# Ubuntu 24.04). BUCKET and TOPIC_ARN must be exported at the top of the file.
# It pulls h100/ and bench/ from S3, runs run_round2.sh, uploads its own log and shuts the instance down
# (launch with --instance-initiated-shutdown-behavior terminate). Keep it small: EC2 user data is limited to 16 KB.

WORK=${WORK:-/opt/nemotron/work}
HARD_CAP_MIN=${HARD_CAP_MIN:-600}
exec >> /var/log/nemotron-h100-round2.log 2>&1
shutdown -h +"$HARD_CAP_MIN" "nemotron round 2: ${HARD_CAP_MIN} min cap" || true     # dead-man switch

echo "[$(date -u +%FT%TZ)] h100/user-data-round2.sh start"
: "${BUCKET:?BUCKET not set}"
export BUCKET TOPIC_ARN AWS_REGION="${AWS_REGION:-us-east-2}" AWS_DEFAULT_REGION="${AWS_REGION:-us-east-2}"
export WORK HARD_CAP_MIN

notify() { [ -n "${TOPIC_ARN:-}" ] && aws sns publish --topic-arn "$TOPIC_ARN" --subject "$(printf '%s' "$1" | tr -d '\r\n' | cut -c1-99)" --message "${2:-none}" >/dev/null 2>&1; return 0; }

command -v aws >/dev/null 2>&1 || snap install aws-cli --classic || { apt-get update -y && apt-get install -y awscli; }
ok=0
for i in 1 2 3 4 5 6; do     # the instance profile credentials can take a few seconds to appear
  mkdir -p "$WORK" && cd "$WORK" &&
    aws s3 sync --only-show-errors "s3://$BUCKET/work/h100/" h100/ &&
    aws s3 sync --only-show-errors "s3://$BUCKET/work/bench/" bench/ && { ok=1; break; }
  sleep 20
done
if [ "$ok" != 1 ] || [ ! -f "$WORK/h100/run_round2.sh" ]; then
  notify "[nemotron] H100 round 2 FAILED: could not fetch run_round2.sh" "aws s3 sync from s3://$BUCKET/work/h100/ and bench/ failed after 6 tries. Shutting down."
  aws s3 cp --only-show-errors /var/log/nemotron-h100-round2.log "s3://$BUCKET/results/h100/round2/logs/user-data-failed.log" || true
  shutdown -h now
  exit 1
fi
chmod +x "$WORK"/h100/*.sh
IID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')" http://169.254.169.254/latest/meta-data/instance-id)
notify "[nemotron] H100 round 2 instance up ($IID)" "Starting the round-2 run at $(date -u +%FT%TZ). $(nvidia-smi -L 2>/dev/null | head -8)"

cd "$WORK" || exit 1
export RUN_ALL_NO_SHUTDOWN=1      # this script shuts the instance down
bash "$WORK/h100/run_round2.sh"
rc=$?
echo "[$(date -u +%FT%TZ)] run_round2.sh exit $rc"
[ "$rc" != 0 ] && notify "[nemotron] H100 round 2: run_round2.sh exited with $rc" "See s3://$BUCKET/results/h100/round2/logs/. The instance shuts down now."
aws s3 cp --only-show-errors /var/log/nemotron-h100-round2.log "s3://$BUCKET/results/h100/round2/logs/user-data-${IID:-unknown}.log" || true
shutdown -h now
exit "$rc"
