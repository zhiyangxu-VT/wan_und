#!/bin/bash

# Set data directory - change this to your dataset location
DATA_DIR="/path_to_BLIP3o-NEXT-EDIT-ENSEMBLE_directory"

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=16556 \
    blip3o/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path BLIP3o/BLIP3o-NEXT-SFT-3B \
    --diffusion_name_or_path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --version qwen_1_5 \
    --dataset_cls mix \
    --dispatch_batches False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --data_list $DATA_DIR/nano_banana/Nano-150k/json/Times-Change_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/action_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/background_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/black_headshot_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/hairstyle_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/sweet_headshot_cleaned.jsonl,$DATA_DIR/nano_banana/Nano-150k/json/reconstruct/Times-Change_cleaned_reconstruct_v2.jsonl,$DATA_DIR/nano_banana/N
    --data_list_weights 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,1,1,1,10,10,2,2 \
    --run_name discrete_x2i_2node_5e-5_image_id_only_reproduce \
    --output_dir /fsx/sfr/data/lxue/training_ckpts/blip3o_next/discrete_x2i_2node_5e-5_image_id_only_reproduce \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 4000 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True \
    --use_und_image_vae False \
    --dataloader_pin_memory False \
    --use_und_image_vae_as_noise False \
    --only_use_und_image_vae_as_noise True
