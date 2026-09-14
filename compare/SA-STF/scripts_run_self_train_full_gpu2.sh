#!/usr/bin/env bash
set -euo pipefail

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hf

cd /home/zhaojiazhen/workspace/STF/STF

python compare/SA-STF/run_syy_setting9.py \
  --dataset CIA \
  --split full \
  --device cuda:2 \
  --weight-path compare/SA-STF/train_runs/syy_setting-9/CIA/checkpoints/final.pt \
  --result-root compare/SA-STF/results/syy_setting-9_self_train

python compare/SA-STF/run_syy_setting9.py \
  --dataset LGC \
  --split full \
  --device cuda:2 \
  --weight-path compare/SA-STF/train_runs/syy_setting-9/LGC/checkpoints/final.pt \
  --result-root compare/SA-STF/results/syy_setting-9_self_train
