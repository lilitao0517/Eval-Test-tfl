#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1.5-7b"
################## VICUNA ##################

deepspeed --master_port=$((RANDOM + 10000)) --include localhost:2,3 stingbee/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 64 --lora_alpha 16 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path path/to/base/llavav1.5-7b \
    --version $PROMPT_VERSION \
    --data_path path/to/StingBee_XrayInstruct.json \
    --image_folder path/to/Images_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter path/to/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir path/to/checkpoints_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 7000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
    --report_to wandb
    
