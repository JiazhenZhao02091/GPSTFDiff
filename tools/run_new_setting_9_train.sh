#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_CONDA_ENV="${HF_CONDA_ENV:-hf}"
STFMAMBA_CONDA_ENV="${STFMAMBA_CONDA_ENV:-stf}"
LOG_DIR="${LOG_DIR:-logs/new_setting-9/train/$(date +%Y%m%d_%H%M%S)}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Override like:
#   GPU_IDS="0 1 2 3 4 5 6 7 8" ./run_new_setting_9_train.sh
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_IDS is empty"
  exit 1
fi

mkdir -p "$LOG_DIR"

pids=()
names=()

gpu_for() {
  local idx="$1"
  echo "${GPUS[$((idx % ${#GPUS[@]}))]}"
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing file: $path"
    exit 1
  fi
}

run_entry() {
  local conda_env="$1"
  local entry="$2"
  local config="$3"
  "$CONDA_BIN" run --no-capture-output -n "$conda_env" "$PYTHON_BIN" - "$entry" "$config" <<'PY'
import re
import sys
from pathlib import Path

entry = sys.argv[1]
config = sys.argv[2]
sys.argv = [entry, "--congfig_path", config]

source = Path(entry).read_text(encoding="utf-8")
source = re.sub(
    r"(?m)^(\s*)os\.environ\[['\"]CUDA_VISIBLE_DEVICES['\"]\]\s*=\s*['\"][^'\"]*['\"]\s*(#.*)?$",
    r"\1# CUDA_VISIBLE_DEVICES is set by run_new_setting_9_train.sh",
    source,
)

code = compile(source, entry, "exec")
exec(code, {"__name__": "__main__", "__file__": entry, "__package__": None})
PY
}

launch_one() {
  local name="$1"
  local gpu="$2"
  local conda_env="$3"
  local entry="$4"
  local config="$5"
  local log="$LOG_DIR/${name}.log"

  check_file "$entry"
  check_file "$config"

  echo "[START] $name env=$conda_env gpu=$gpu log=$log"
  (
    set -e
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS
    echo "[START] $(date '+%F %T') $name"
    echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[CONDA] $conda_env"
    echo "[CMD] $CONDA_BIN run --no-capture-output -n $conda_env $PYTHON_BIN $entry --congfig_path $config"
    run_entry "$conda_env" "$entry" "$config"
    echo "[DONE] $(date '+%F %T') $name"
  ) > "$log" 2>&1 &

  pids+=("$!")
  names+=("$name")
}

launch_two_stage() {
  local name="$1"
  local gpu="$2"
  local conda_env="$3"
  local entry="$4"
  local stage1_config="$5"
  local stage2_config="$6"
  local log="$LOG_DIR/${name}.log"

  check_file "$entry"
  check_file "$stage1_config"
  check_file "$stage2_config"

  echo "[START] $name env=$conda_env gpu=$gpu log=$log"
  (
    set -e
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS
    echo "[START] $(date '+%F %T') $name stage_1"
    echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[CONDA] $conda_env"
    echo "[CMD] $CONDA_BIN run --no-capture-output -n $conda_env $PYTHON_BIN $entry --congfig_path $stage1_config"
    run_entry "$conda_env" "$entry" "$stage1_config"
    echo "[DONE] $(date '+%F %T') $name stage_1"
    echo "[START] $(date '+%F %T') $name stage_2"
    echo "[CMD] $CONDA_BIN run --no-capture-output -n $conda_env $PYTHON_BIN $entry --congfig_path $stage2_config"
    run_entry "$conda_env" "$entry" "$stage2_config"
    echo "[DONE] $(date '+%F %T') $name stage_2"
  ) > "$log" 2>&1 &

  pids+=("$!")
  names+=("$name")
}

# launch_one "fsdformer" "$(gpu_for 0)" \
#   "$HF_CONDA_ENV" \
#   "tools/train/train_fsdformer.py" \
#   "config/fsdformer/new_setting-9/ML/train_config.py"

# launch_one "ganstfm" "$(gpu_for 1)" \
#   "$HF_CONDA_ENV" \
#   "tools/train/train_ganstfm.py" \
#   "config/ganstfm/new_setting-9/ML/Adam_1e-4.py"

launch_one "opgan" "$(gpu_for 0)" \
  "$HF_CONDA_ENV" \
  "tools/train/train_opgan.py" \
  "config/opgan/new_setting-9/ML/RMSProp~w_change_1e-2.py"

launch_one "GPSTFDiff" "$(gpu_for 1)" \
  "$HF_CONDA_ENV" \
  "tools/train/train_GPSTFDiff.py" \
  "config/GPSTFDiff/new_setting_9/ML/config_MKIRA_mult_4_dim_128.py"

launch_two_stage "stfdcnn" "$(gpu_for 2)" \
  "$HF_CONDA_ENV" \
  "tools/train/train_stfdcnn.py" \
  "config/stfdcnn/new_setting~9/ML/stage_1_SGD_1e-2.py" \
  "config/stfdcnn/new_setting~9/ML/stage_2_SGD_1e-1.py"

# launch_one "stfdiff" "$(gpu_for 5)" \
#   "$HF_CONDA_ENV" \
#   "tools/train/train_stfdiff.py" \
#   "config/stfdiff/new_setting-9/model6_GN_SiLU/ML/config.py"

launch_two_stage "stfgan" "$(gpu_for 3)" \
  "$HF_CONDA_ENV" \
  "tools/train/train_stfgan.py" \
  "config/stfgan/new_setting-9/ML/stage_1~RMSProp.py" \
  "config/stfgan/new_setting-9/ML/stage_2~RMSProp.py"

launch_one "stfmamba" "$(gpu_for 4)" \
  "$STFMAMBA_CONDA_ENV" \
  "tools/train/train_stfmamba.py" \
  "config/stfmamba/new_setting-9/ML/config.py"

launch_one "swinstf" "$(gpu_for 5)" \
  "$HF_CONDA_ENV" \
  "tools/train/train_swinstf.py" \
  "config/swinstf/new_setting-9/ML/Adam_1e-4~StepLR_step_size-15_gamma_5e-1.py"

echo "[INFO] launched ${#pids[@]} training jobs"
echo "[INFO] logs: $LOG_DIR"

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[OK] ${names[$i]}"
  else
    echo "[FAIL] ${names[$i]} log=$LOG_DIR/${names[$i]}.log"
    status=1
  fi
done

exit "$status"
