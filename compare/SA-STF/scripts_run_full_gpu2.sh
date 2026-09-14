#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaojiazhen/workspace/STF/STF
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hf
export PYTHONUNBUFFERED=1

python compare/SA-STF/run_syy_setting9.py \
  --dataset all \
  --split full \
  --device cuda:2 \
  --result-root compare/SA-STF/results/syy_setting-9
