#!/usr/bin/env bash
set -euo pipefail

COMPARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COMPARE_ROOT"

LOG_DIR="SA-STF/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/ml_scale1_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec >> "$PIPELINE_LOG" 2>&1

echo "[start] $(date '+%F %T')"
echo "[cwd] $COMPARE_ROOT"
echo "[log] $PIPELINE_LOG"

DEVICE="cuda:5"
TRAIN_ROOT="SA-STF/train_runs/syy_setting-9_ML_scale1"
TRAIN_SUMMARY="$TRAIN_ROOT/ML/train_summary.json"
WEIGHT="$TRAIN_ROOT/ML/checkpoints/final.pt"
SELF_RESULT_ROOT="SA-STF/results/syy_setting-9_ML_self_train_scale1"
OFFICIAL_RESULT_ROOT="SA-STF/results/syy_setting-9_ML_official_AHB_weight_scale1"
CLASSIFIER_ROOT="SA-STF/classifier_eval/ML_syy_setting-9_scale1"

while [[ ! -f "$TRAIN_SUMMARY" ]] && pgrep -f "SA-STF/train_syy_setting9.py --dataset ML --output-root $TRAIN_ROOT" >/dev/null; do
  echo "[train] existing foreground/background ML scale1 training detected, wait 60s"
  sleep 60
done

if [[ ! -f "$TRAIN_SUMMARY" ]]; then
  echo "[train] ML scale1 training starts on $DEVICE"
  conda run --no-capture-output -n hf python SA-STF/train_syy_setting9.py \
    --dataset ML \
    --output-root "$TRAIN_ROOT" \
    --device "$DEVICE" \
    --epochs 200 \
    --batch-size 4 \
    --num-workers 4 \
    --save-every 20 \
    --perceptual vgg
else
  echo "[train] found existing $TRAIN_SUMMARY, skip training"
fi

if [[ ! -f "$WEIGHT" ]]; then
  echo "[error] missing final checkpoint: $WEIGHT" >&2
  exit 1
fi

echo "[infer] self-trained full"
conda run --no-capture-output -n hf python SA-STF/run_syy_setting9.py \
  --dataset ML \
  --split full \
  --result-root "$SELF_RESULT_ROOT" \
  --weight-path "$WEIGHT" \
  --device "$DEVICE"

echo "[infer] self-trained patch"
conda run --no-capture-output -n hf python SA-STF/run_syy_setting9.py \
  --dataset ML \
  --split patch \
  --result-root "$SELF_RESULT_ROOT" \
  --weight-path "$WEIGHT" \
  --device "$DEVICE"

echo "[infer] AHB official weight on ML full"
conda run --no-capture-output -n hf python SA-STF/run_syy_setting9.py \
  --dataset ML \
  --split full \
  --result-root "$OFFICIAL_RESULT_ROOT" \
  --weight-path SA-STF/weights/ckpt_SA_STF_AHB.pt \
  --device "$DEVICE"

echo "[infer] AHB official weight on ML patch"
conda run --no-capture-output -n hf python SA-STF/run_syy_setting9.py \
  --dataset ML \
  --split patch \
  --result-root "$OFFICIAL_RESULT_ROOT" \
  --weight-path SA-STF/weights/ckpt_SA_STF_AHB.pt \
  --device "$DEVICE"

echo "[metric] per-band metrics"
conda run --no-capture-output -n hf python SA-STF/compute_band_metrics.py

echo "[classifier] self-trained full/patch"
conda run --no-capture-output -n hf python SA-STF/classifier_eval_sastf.py \
  --method-name SA-STF_self_train_scale1_final \
  --mode both \
  --full-pred-dir "$SELF_RESULT_ROOT/ML/full/imgs/ML/0/save_img" \
  --patch-pred-dir "$SELF_RESULT_ROOT/ML/patch/imgs/ML/0/save_img" \
  --output-root "$CLASSIFIER_ROOT" \
  --device "$DEVICE"

echo "[classifier] AHB official full/patch"
conda run --no-capture-output -n hf python SA-STF/classifier_eval_sastf.py \
  --method-name SA-STF_official_AHB_weight_scale1 \
  --mode both \
  --full-pred-dir "$OFFICIAL_RESULT_ROOT/ML/full/imgs/ML/0/save_img" \
  --patch-pred-dir "$OFFICIAL_RESULT_ROOT/ML/patch/imgs/ML/0/save_img" \
  --output-root "$CLASSIFIER_ROOT" \
  --device "$DEVICE"

echo "[report] update generated markdown report"
conda run --no-capture-output -n hf python SA-STF/update_ml_scale1_report.py

echo "[done] $(date '+%F %T')"
