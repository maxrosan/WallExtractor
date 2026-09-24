#!/usr/bin/env bash
# Runs inside the RunPod pod. Started by scripts/runpod_ctl.py through dockerStartCmd.
# Everything it prints goes to /workspace/job.log, which the controller reads
# over the Jupyter contents API (HTTPS only; SSH is not needed).
#
# Env (set by the controller):
#   WE_BRANCH      git branch to run (default main)
#   WE_TRAIN/WE_VAL number of plans to fetch
#   WE_EPOCHS, WE_SIZE, WE_BATCH, WE_MODEL, WE_MAX_MINUTES  training knobs
#   WE_IDLE_MINUTES minutes to keep the pod alive after the job for follow-up commands
#   RUNPOD_API_KEY  lets the pod terminate itself when idle (budget guard)
set -o pipefail
export PYTHONUNBUFFERED=1
cd /workspace || exit 1
REPO=https://github.com/maxrosan/WallExtractor.git
BRANCH=${WE_BRANCH:-main}
mkdir -p /workspace/results
echo "[job] start $(date -u +%FT%TZ) branch=$BRANCH gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

self_terminate() {
  echo "[job] terminating pod $RUNPOD_POD_ID ($1)"
  if [ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ]; then
    curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || true
  fi
}

# Hard budget guard: whatever happens, the pod dies after WE_HARD_MINUTES.
( sleep $(( ${WE_HARD_MINUTES:-150} * 60 )); self_terminate "hard limit" ) &

if [ ! -d /workspace/we ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO" /workspace/we || { echo "[job] clone failed"; sleep 600; self_terminate "clone failed"; exit 1; }
fi
cd /workspace/we
pip install -q -r requirements-train.txt 2>&1 | tail -3
python -c "import torch, transformers; print('[job] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

python scripts/fetch_cubicasa_subset.py --out /workspace/data/cubicasa5k --train "${WE_TRAIN:-400}" --val "${WE_VAL:-100}" \
  && python -m wallextractor.cubicasa --root /workspace/data/cubicasa5k --out /workspace/data/prepared \
  && python -m wallextractor.train_seg --data /workspace/data/prepared --out /workspace/results \
       --epochs "${WE_EPOCHS:-15}" --size "${WE_SIZE:-512}" --batch "${WE_BATCH:-8}" --model "${WE_MODEL:-nvidia/mit-b1}" \
       --max-minutes "${WE_MAX_MINUTES:-45}" --export-onnx
echo "[job] JOB_DONE rc=$? $(date -u +%FT%TZ)"

# Command loop: the controller uploads /workspace/cmd.sh; output goes to /workspace/cmd.log.
idle_deadline=$(( $(date +%s) + ${WE_IDLE_MINUTES:-30} * 60 ))
while [ "$(date +%s)" -lt "$idle_deadline" ]; do
  if [ -f /workspace/cmd.sh ]; then
    mv /workspace/cmd.sh /workspace/cmd.running.sh
    echo "[job] running cmd $(date -u +%FT%TZ)"
    bash /workspace/cmd.running.sh > /workspace/cmd.log 2>&1
    echo "[job] CMD_DONE rc=$?" >> /workspace/cmd.log
    idle_deadline=$(( $(date +%s) + ${WE_IDLE_MINUTES:-30} * 60 ))
  fi
  sleep 5
done
self_terminate "idle"
