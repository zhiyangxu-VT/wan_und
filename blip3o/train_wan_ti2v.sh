#!/bin/bash

set -euo pipefail
module load cuda12.6/toolkit/12.6.2
conda activate wan

# Paths
export HF_HOME=/projects/nlp_lab/zhiyang/phd6_projects/hf_hub
export OUTPUT_FOLDER=/Your/Model/Output/
export WAN_MODEL_PATH=/path/to/Wan-AI/Wan2.2-TI2V-5B
export VIDEO_METADATA=/path/to/video_train.jsonl
export VIDEO_ROOT=/path/to/frames_root
export WANDB_API_KEY='fc775ec821d5a66ffa21a840f216e8fba0adbbca'
# Example JSONL line:
# {"id":"sample-1","prompt":"A robot walking in the snow.","video":"videos/sample-1","input_image":"videos/sample-1/000000.png"}

# NOTE: WAN backend requires Qwen backbone (vision_tower must be None).

torchrun --nproc_per_node=8 \
  /projects/nlp_lab/zhiyang/final_projects/video_und/wan_und/blip3o/train/train_mem.py \
  --deepspeed /projects/nlp_lab/zhiyang/final_projects/video_und/wan_und/deepspeed_scripts/zero1.json \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --version qwen \
  --data_type video \
  --video_metadata_path ${VIDEO_METADATA} \
  --video_folder ${VIDEO_ROOT} \
  --video_diffusion_backend wan2.2-ti2v-5b \
  --wan_model_path ${WAN_MODEL_PATH} \
  --train_wan_dit False \
  --freeze_backbone False \
  --gen_vision_tower eva-clip-E-14-plus \
  --gen_projector_type mlp2x_gelu \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ${OUTPUT_FOLDER} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.003 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --lr_scheduler_kwargs '{"min_lr":1e-5}' \
  --model_max_length 512 \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --gen_pooling early_pool2d_4 \
  --n_query 64 \
  --n_und_query 0 \
  --report_to wandb \
  --run_name blip3o_qwen_wan_ti2v
