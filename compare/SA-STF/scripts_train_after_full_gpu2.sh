#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaojiazhen/workspace/STF/STF

while kill -0 "$(cat compare/SA-STF/results/syy_setting-9/run_full_gpu2.pid)" 2>/dev/null; do
  sleep 60
done

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hf
export PYTHONUNBUFFERED=1

python compare/SA-STF/train_syy_setting9.py \
  --dataset all \
  --device cuda:2 \
  --output-root compare/SA-STF/train_runs/syy_setting-9 \
  --perceptual vgg
