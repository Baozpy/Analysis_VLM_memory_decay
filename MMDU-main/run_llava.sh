#!/bin/bash
#SBATCH --job-name=llava_full
#SBATCH --account=cml-furongh
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llava_%j.out
#SBATCH --error=logs/llava_%j.err

# -------- 环境准备 (和你平时手动做的一样) --------
source /nfshomes/byan1/venvs/cmsc799/bin/activate

export SCRATCH=/fs/nexus-scratch/$USER
export HF_HOME=$SCRATCH/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=$SCRATCH/tmp

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

# 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# -------- 真正跑 --------
python -u pilot_multimodel_v1.py \
  --model llava-hf/llava-v1.6-vicuna-7b-hf \
  --start-sample 109 \
  --num-samples 1 \
  --max-turns 20 \
  --max-new-tokens 128 \
  --dtype bf16 \
  --device-map auto \
  --beams 3 \
  --use-history --hist-max 2 \
  --results output/llava_dummy.csv \
  --debug
