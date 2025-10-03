#!/bin/bash

set -e

# Environment setup
export CONDA_ROOT=/fsx/home/jiuhai.chen/envs/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate blip3o

export PATH="/fsx/home/jiuhai.chen/envs/miniconda3/envs/blip3o/bin:$PATH"

INPUT_IMAGE="/fsx/home/lxue/repos/BLIP3o/inference_scripts/images/bird_cropped_square.jpg"
DEVICE="cuda:0"
INSTRUCTION="Change the bird’s feather color to red while keeping the bird’s body, beak, eyes, and the background exactly the same."

# only use image id
MODEL_NAME="discrete_x2i_3node_5e-5_image_id_only_ensemble_large"
MODEL_PATH="/fsx/sfr/data/lxue/training_ckpts/blip3o_next/${MODEL_NAME}/checkpoint-61916"
OUTPUT="editing_inference/${MODEL_NAME}.png"
python editing_inference/inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --only_use_und_image_vae_as_noise

# /fsx/sfr/data/lxue/training_ckpts/blip3o_next/discrete_x2i_2node_5e-5_image_id_only_only_single_input_image_ensemble_large_rerun
MODEL_NAME="discrete_x2i_2node_5e-5_image_id_only_only_single_input_image_ensemble_large_rerun"
MODEL_PATH="/fsx/sfr/data/lxue/training_ckpts/blip3o_next/${MODEL_NAME}/checkpoint-61916"
OUTPUT="editing_inference/${MODEL_NAME}.png"
python editing_inference/inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --only_use_und_image_vae_as_noise

# use image id and vae as cross attention
# /fsx/sfr/data/lxue/training_ckpts/blip3o_next/discrete_x2i_2node_5e-5_image_id_and_cross_attn_only_single_input_image_ensemble_large
MODEL_NAME="discrete_x2i_2node_5e-5_image_id_and_cross_attn_only_single_input_image_ensemble_large"
# fill in the checkpoint number
MODEL_PATH="/fsx/sfr/data/lxue/training_ckpts/blip3o_next/${MODEL_NAME}/checkpoint-"
OUTPUT="editing_inference/${MODEL_NAME}.png"
python editing_inference/inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --only_use_und_image_vae_as_noise \
    --use_und_image_vae \

# use image id and vae as noise concat as height
# /fsx/sfr/data/lxue/training_ckpts/blip3o_next/discrete_x2i_2node_5e-5_image_id_vae_as_height_only_single_input_image_ensemble_large
MODEL_NAME="discrete_x2i_2node_5e-5_image_id_vae_as_height_only_single_input_image_ensemble_large"
MODEL_PATH="/fsx/sfr/data/lxue/training_ckpts/blip3o_next/${MODEL_NAME}/checkpoint-"
OUTPUT="editing_inference/${MODEL_NAME}.png"
python editing_inference/inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --only_use_und_image_vae_as_noise \
    --use_und_image_vae_as_noise \
    --use_und_image_vae

# use image id and vae as noise concat as height and cross attention
# /fsx/sfr/data/lxue/training_ckpts/blip3o_next/discrete_x2i_2node_5e-5_image_id_vae_as_height_and_cross_attention_only_single_input_image_ensemble_large
MODEL_NAME="discrete_x2i_2node_5e-5_image_id_vae_as_height_and_cross_attention_only_single_input_image_ensemble_large"
MODEL_PATH="/fsx/sfr/data/lxue/training_ckpts/blip3o_next/${MODEL_NAME}/checkpoint-"
OUTPUT="editing_inference/${MODEL_NAME}.png"
python editing_inference/inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --use_und_image_vae_as_noise \
    --use_und_image_vae