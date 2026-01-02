#!/usr/bin/env bash
#SBATCH --job-name=blip2_full
#SBATCH --account=cml-furongh
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/blip2_%j.out
#SBATCH --error=logs/blip2_%j.err

set -euo pipefail

# ====== 环境与缓存 ======
source ~/venvs/cmsc799/bin/activate

export SCRATCH=/fs/nexus-scratch/$USER
export HF_HOME=$SCRATCH/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

# 减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# ====== 目录与模型 ======
PROJ=/fs/nexus-scratch/$USER/MMDU-main
cd "$PROJ"

OUT="$PROJ/output"
LOGS="$PROJ/logs"
mkdir -p "$OUT" "$LOGS"

MODEL="Salesforce/blip2-opt-6.7b"
SAFE_MODEL="Salesforce_blip2-opt-6.7b"   # 用于文件名

# ====== 起止样本（含首尾）======
START=97            # 从 sample6 开始
END=109            # 你说总样本是 0-109

echo "Node: $(hostname)"
nvidia-smi || true
echo "Start running $MODEL from sample $START to $END"
echo "Outputs -> $OUT/${SAFE_MODEL}_sample{ID}.csv"
echo

# ====== 循环逐样本评测（容错 + 断点续跑）======
for S in $(seq $START $END); do
  OUT_CSV="$OUT/${SAFE_MODEL}_sample${S}.csv"

  if [[ -s "$OUT_CSV" ]]; then
    echo "[skip] sample $S -> exists: $OUT_CSV"
    continue
  fi

  echo ">>> [run] sample $S ..."
  set +e
  python -u pilot_multimodel_v1.py \
    --model "$MODEL" \
    --start-sample "$S" \
    --num-samples 1 \
    --max-turns 20 \
    --max-new-tokens 128 \
    --dtype bf16 --device-map auto \
    --use-history --hist-max 2 \
    --beams 3 \
    --panel-tile-size 336 \
    --results "$OUT_CSV" \
    --debug
  CODE=$?
  set -e

  if [[ $CODE -ne 0 ]]; then
    echo "[FAIL] sample $S (exit=$CODE) — see logs" | tee -a "$LOGS/blip2_failures.txt"
  else
    echo "[ok]   sample $S -> $OUT_CSV"
  fi

  # 每个样本后小憩一下，给调度/缓存降温（可选）
  sleep 3
done

echo
echo "All done. Failed samples (if any) -> $LOGS/blip2_failures.txt"
