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

DISTILLATOR=modeling_amt.experimental:Distillator
TEACHER_CLS=transformers:AutoModelForCausalLM
DISTIL_ALPHA=1.0
# TASK_NAMES=dolly:babilong_qa1_0k:babilong_qa1_2k:babilong_qa1_4k
# TASK_RATIOS=0.9:0.09:0.05

TASK_NAMES=HuggingFaceTB/smoltalk:babilong_qa1_0k:babilong_qa1_2k:babilong_qa2_0k:babilong_qa2_2k:babilong_qa3_0k:babilong_qa1_2k:babilong_qa3_0k:babilong_qa1_2k:babilong_qa4_0k:babilong_qa4_2k:babilong_qa5_0k:babilong_qa5_2k
TASK_RATIOS=0.1:0.09:0.09:0.09:0.09:0.09:0.09:0.09:0.09:0.09:0.09

ITERS=5000
TBS=64
INPUT_SIZE=1024

MAX_N_SEGMENTSS=(2)
MEMORY_SIZE=16
D_MEM=64
BSS=(4)
for N in 1
do

for MODEL_NAME in "/home/jovyan/kuratov/models/Llama-3.2-1B-Instruct/"
# for MODEL_NAME in "/home/jovyan/kuratov/models/Llama-3.2-1B/"
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
# for ARMT we have memory embs only at the end of the sequence
INPUT_SEQ_LEN=$(((INPUT_SIZE-MEMORY_SIZE)*MAX_N_SEGMENTS))
K2=2
K2_PREV=2
BS=${BSS[j]}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in 3e-04 1e-05
do

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM
accelerate launch --num_processes $NP --config_file ./accelerate_bf16.yaml run_finetuning_dataset_mix_armt_distill.py \
        --task_names $TASK_NAMES \
        --task_ratios $TASK_RATIOS \
        --output_dir /home/jovyan/rmt/runs/test/${TASK_NAME}/Llama-3.2-1B-Instruct/smol:qa1-5-1:9/SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_LR${LR}-lora-mnc-distill_${DISTIL_ALPHA}_short \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --distillator_cls $DISTILLATOR \
        --teacher_cls $TEACHER_CLS \
        --pretrained_teacher $MODEL_NAME \
        --alpha_distil $DISTIL_ALPHA \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --layers_attr base_model.base_model.layers \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 2 \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.001 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 3000 \
        --data_n_workers 2 \
        --logging_steps 5 --eval_steps 10 --save_steps 50 \
        --show_valid_examples 0 \
        --tokenizer_for_chat_template "/home/jovyan/kuratov/models/Llama-3.2-1B-Instruct/" \
        --early_stopping_patience 75 \
        --seed $(($N+42)) \
        --max_grad_norm 1.0 \
        --d_mem $D_MEM \
        --no_packing \
        --use_length_filtering \
        --mask_non_completion \
        --use_lora \
        --report_to tensorboard
done
done
done
done
done
done
echo "done"
