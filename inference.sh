#!/bin/bash

INPUT_IMAGE="blue_bird.jpg"
DEVICE="cuda:0"
INSTRUCTION="Change the bird’s feather color to red while keeping the bird’s body, beak, eyes, and the background exactly the same."

# --use_und_image_vae  means using as cross-attention inputs
# --use_und_image_vae_as_noise means VAE as noise-space injection
MODEL_NAME="cross_attn_noise"
MODEL_PATH="your model path"
python inference_editing.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IMAGE" \
    --instruction "$INSTRUCTION" \
    --output "output.png" \
    --device "$DEVICE" \
    --use_und_image_vae \
    --use_und_image_vae_as_noise
