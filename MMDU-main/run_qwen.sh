#!/bin/bash
#SBATCH --job-name=qwen_full
#SBATCH --account=cml-furongh
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err

# -------- 环境准备 --------
source /nfshomes/byan1/venvs/cmsc799/bin/activate

export SCRATCH=/fs/nexus-scratch/$USER
export HF_HOME=$SCRATCH/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=$SCRATCH/tmp

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# -------- 真正跑 --------
python -u pilot_multimodel_v1.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --start-sample 108 \
  --num-samples 110 \
  --max-turns 20 \
  --max-new-tokens 128 \
  --dtype bf16 \
  --device-map auto \
  --beams 3 \
  --use-history --hist-max 2 \
  --results output/qwen_dummy.csv \
  --debug
