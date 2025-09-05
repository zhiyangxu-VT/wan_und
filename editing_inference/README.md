# BLIP3o Editing Model Inference

This directory contains a functional inference script for the BLIP3o editing model that supports both text-to-image generation and image editing.

## Files

- `inference_editing.py` - Main inference script
- `test_editing.py` - Test script for validation
- `README.md` - This documentation

## Setup

Use your existing conda environment:

```bash
export CONDA_ROOT=/fsx/home/jiuhai.chen/envs/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate blip3o
```

## Usage

### Text-to-Image Generation

```bash
python inference_editing.py \
  --instruction "A beautiful sunset over mountains" \
  --output "sunset.png"
```

### Image Editing

```bash
python inference_editing.py \
  --instruction "Change the color to blue" \
  --input_image "path/to/input/image.jpg" \
  --output "edited_image.png"
```

### Parameters

- `--model_path, -m`: Path to the trained editing model (default: your checkpoint-2000)
- `--input_image, -i`: Path to input image for editing (optional, for text-to-image if not provided)
- `--instruction, -t`: Text instruction for generation/editing (required)
- `--output, -o`: Output image path (optional, auto-generated if not provided)
- `--device, -d`: Device to use (default: cuda:0)

## Model Architecture Support

The script properly handles:

1. **Understanding Images**: Input images for editing are processed through the vision tower and integrated as understanding tokens
2. **Generation Pipeline**: Uses the trained editing model's diffusion components for image generation
3. **Token Handling**: Correctly processes special tokens for vision understanding and image generation
4. **Training Format Compatibility**: Matches the exact conversation template used during training

## Features

- ✅ Text-to-image generation
- ✅ Image editing with input images
- ✅ Proper understanding image processing
- ✅ Compatible with the current codebase architecture
- ✅ Handles the trained editing model checkpoint
- ✅ Supports multiple input images (up to 4, as per training data)

## Example Outputs

The script will save generated/edited images in the current directory with descriptive filenames based on the instruction text.

## Troubleshooting

If you encounter CUDA memory issues, try reducing the sequence length in the EditingConfig class or using a smaller guidance scale for the diffusion process.