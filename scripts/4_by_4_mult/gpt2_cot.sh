#!/usr/bin/env bash
set -e
cd ../../
# NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES="0"
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM

TASK_NAME=4_by_4_mult

ITERS=25000
TBS=64
INPUT_SIZE=1024

for N in 1
do

MODEL_NAME=gpt2
SEGMENT_ORDERING=regular
MAX_N_SEGMENTS=1
BS=32

SCHEDULER=linear
INPUT_SEQ_LEN=$(((INPUT_SIZE)))

for LR in 3e-04 1e-05
do

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
ACCEL_CONFIG="/workspace-SR006.nfs2/Bulatov_A/rmt/reasoning/accel_configs/accelerate_bf16.yaml"


echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM
accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG run_finetuning_reasoning_rmt.py \
        --task_name $TASK_NAME \
        --dataset_name "booydar/4_by_4_mult" \
        --output_dir /workspace-SR006.nfs2/Bulatov_A/rmt/runs/${TASK_NAME}/${MODEL_NAME}/SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_LR${LR}-cot \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_cot \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --layers_attr base_model.base_model.layers \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --k1 -1 --k2 -1 \
        --optimizer AdamW  --weight_decay 0.001 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 3000 \
        --data_n_workers 2 \
        --logging_steps 50 --eval_steps 250 --save_steps 500 \
        --show_valid_examples 0 \
        --early_stopping_patience 75 \
        --seed $(($N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to tensorboard
done
done
done
done
done
done
echo "done"
