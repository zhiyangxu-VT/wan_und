#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_DATASET_ROOT="${WORKSPACE_ROOT}/DiffSynth-Studio/data/example_video_dataset"
VIDEO_FRAMES_ROOT="${VIDEO_DATASET_ROOT}/frames"
VIDEO_METADATA="${VIDEO_DATASET_ROOT}/video_train.jsonl"

export OUTPUT_FOLDER=./models
export WAN_MODEL_PATH="${WORKSPACE_ROOT}/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B"
export VIDEO_DATASET_ROOT
export VIDEO_FRAMES_ROOT
export VIDEO_METADATA

cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

WAN_VAE_FALLBACK="${WORKSPACE_ROOT}/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
if [[ ! -f "${WAN_MODEL_PATH}/Wan2.2_VAE.safetensors" && -f "${WAN_VAE_FALLBACK}" ]]; then
  ln -sf "${WAN_VAE_FALLBACK}" "${WAN_MODEL_PATH}/Wan2.2_VAE.safetensors"
fi

mkdir -p "${VIDEO_FRAMES_ROOT}"

if [[ ! -d "${VIDEO_FRAMES_ROOT}/video1" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found; install it or pre-extract frames under ${VIDEO_FRAMES_ROOT}." >&2
    exit 1
  fi
  mkdir -p "${VIDEO_FRAMES_ROOT}/video1"
  ffmpeg -y -hide_banner -loglevel error \
    -i "${VIDEO_DATASET_ROOT}/video1.mp4" \
    "${VIDEO_FRAMES_ROOT}/video1/%06d.png"
fi

if [[ ! -f "${VIDEO_METADATA}" ]]; then
  python - <<'PY'
import csv
import json
import os

root = os.environ["VIDEO_DATASET_ROOT"]
frames_root = os.environ["VIDEO_FRAMES_ROOT"]
out_path = os.environ["VIDEO_METADATA"]
meta_path = os.path.join(root, "metadata.csv")

with open(meta_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit("metadata.csv is empty")

with open(out_path, "w", encoding="utf-8") as out:
    for idx, row in enumerate(rows):
        video = row["video"]
        prompt = row.get("prompt", "")
        name = os.path.splitext(os.path.basename(video))[0]
        rel_frame_dir = os.path.join("frames", name)
        sample = {
            "id": f"sample-{idx + 1}",
            "prompt": prompt,
            "video": rel_frame_dir,
            "input_image": os.path.join(rel_frame_dir, "000001.png"),
        }
        out.write(json.dumps(sample, ensure_ascii=True) + "\n")

print(f"Wrote {out_path}")
PY
fi

export VIDEO_ROOT="${VIDEO_DATASET_ROOT}"
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

torchrun --nproc_per_node=${NPROC_PER_NODE} \
  -m blip3o.train.train_mem \
  --deepspeed ./deepspeed_scripts/zero1.json \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --version qwen \
  --data_type video \
  --video_metadata_path ${VIDEO_METADATA} \
  --video_folder ${VIDEO_ROOT} \
  --video_diffusion_backend wan2.2-ti2v-5b \
  --wan_model_path ${WAN_MODEL_PATH} \
  --train_wan_dit False \
  --freeze_backbone False \
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
  --model_max_length 512 \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --n_query 64 \
  --n_und_query 0 \
  --report_to none \
  --run_name blip3o_qwen_wan_ti2v_sample
