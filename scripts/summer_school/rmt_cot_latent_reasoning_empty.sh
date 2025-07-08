#!/usr/bin/env bash
set -e

SCRIPT_DIR=/home/user30/booydar/deep-reasoning
RUNS_DIR=/home/user30/kashurin/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.experimental:RecurrentWrapperNoSegmentation
BACKBONE_CLS=transformers:AutoModelForCausalLM

TASK_NAME=gsm8k
N_EPOCHS=25
GRADIENT_ACC_STEP=8   # should be 8
BS=64
INPUT_SEQ_LEN=64
# MODEL_CPT=/home/user30/kashurin/checkpoint-29500/pytorch_model.bin
MODEL_ID=HuggingFaceTB/SmolLM2-135M
FULL_MODEL_NAME=RMT_SmolLM2-135M
SEGMENT_ORDERING=regular
MAX_N_SEGMENTS=10
MEMORY_SIZE=16
SCHEDULER=constant

ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_fp32_stage2.yaml"
MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_reasoning_rmt-v2.py"


REASONING_MODE=2stage_token_latent_empty
for N in 1; do
    for LR in 3e-04; do
        echo "RUNNING: TASK_NAME SRC_LEN FULL_MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N"
        echo "RUNNING: $TASK_NAME $SRC_LEN $FULL_MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM"

        accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG $MAIN_SCRIPT \
        --task_name $TASK_NAME \
        --dataset_name "booydar/gsm8k" \
        --output_dir ${RUNS_DIR}/${TASK_NAME}/${FULL_MODEL_NAME}/${MAX_N_SEGMENTS}x${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}_BS${BS}_LR${LR}-cot-sft_mode_${REASONING_MODE} \
        --from_pretrained $MODEL_ID \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $INPUT_SEQ_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --max_cot_steps $((MAX_N_SEGMENTS - 2)) \
        --use_cot \
        --reasoning_mode $REASONING_MODE \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --num_train_epochs $N_EPOCHS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --k1 -1 --k2 -1 \
        --optimizer AdamW --weight_decay 0.001 \
        --learning_rate ${LR} \
        --lr_scheduler_type $SCHEDULER \
        --warmup_steps 500 \
        --data_n_workers 2 \
        --logging_steps 10 --eval_steps 50 --save_steps 50 \
        --eval_strategy steps \
        --eval_accumulation_steps 8 \
        --show_valid_examples 0 \
        --early_stopping_patience 75 \
        --seed $((N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to wandb \
        --log_level info
    done
done

echo "done"