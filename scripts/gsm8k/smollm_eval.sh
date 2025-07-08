#!/usr/bin/env bash
set -e

SCRIPT_DIR=/home/user33/kashurin/deep-reasoning
RUNS_DIR=/home/user33/kashurin/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

BACKBONE_CLS=transformers:AutoModelForCausalLM

export CUDA_VISIBLE_DEVICES="3"
NP=1
TASK_NAME=gsm8k
CHECKPOINT=/home/user33/kashurin/TR_SmolLM2-135M/cot/checkpoint-1500/pytorch_model.bin
MODEL_ID=HuggingFaceTB/SmolLM2-135M
MODEL_NAME=SmolLM2-135M
MAX_COT_STEPS=8
MAX_NEW_TOKENS=100

BATCH_SIZE=16
SEED=42

EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluate_reasoning-v2.py"

echo "Evaluating model $CHECKPOINT on $TASK_NAME with max_cot_steps $MAX_COT_STEPS"

python $EVAL_SCRIPT \
    --model_cpt $CHECKPOINT \
    --from_pretrained $MODEL_ID \
    --dataset_name "booydar/gsm8k" \
    --task_name $TASK_NAME \
    --max_cot_steps $MAX_COT_STEPS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --seed $SEED

echo "Evaluation done" 