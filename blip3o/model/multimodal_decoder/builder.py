from diffusers import AutoencoderDC, SanaTransformer2DModel
import torch
import torch.nn as nn

def _initialize_mismatched_parameters(model: SanaTransformer2DModel, initialization: str = "xavier"):
    """Initialize parameters that couldn't be loaded from pretrained weights."""
    initialized_params = []

    for name, param in model.named_parameters():
        # if it contains proj_out or patch_embed, then initialize it.
        if "proj_out" in name or "patch_embed" in name:
            if initialization == "xavier" and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
                initialized_params.append(name)
            elif initialization == "normal":
                nn.init.normal_(param, std=0.02)
                initialized_params.append(name)
            elif initialization == "zeros":
                nn.init.zeros_(param)
                initialized_params.append(name)

    if initialized_params:
        print(f"Re-initialized {len(initialized_params)} parameters with {initialization} initialization")


def build_sana(vision_tower_cfg, **kwargs):
    use_und_image_vae_as_noise = getattr(vision_tower_cfg, "use_und_image_vae_as_noise", False)
    
    if use_und_image_vae_as_noise:
        in_channels = 64
        print(f"Building Sana DiT with {in_channels} input channels...")

        # Load original config and state_dict only once
        try:
            # Load config and state dict manually without creating model
            from huggingface_hub import hf_hub_download
            import os
            
            # Download config and state dict files manually
            config_path = hf_hub_download(
                repo_id=vision_tower_cfg.diffusion_name_or_path,
                filename="config.json",
                subfolder="transformer"
            )
            
            # Load config manually
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            print(f"Original config in_channels: {config_dict.get('in_channels', 'not found')}")
            
            # Modify config
            config_dict['in_channels'] = in_channels
            config_dict['out_channels'] = in_channels
            
    #         # Load state dict manually - try both .safetensors and .bin
    #         try:
    #             state_dict_path = hf_hub_download(
    #                 repo_id=vision_tower_cfg.diffusion_name_or_path,
    #                 filename="diffusion_pytorch_model.safetensors",
    #                 subfolder="transformer"
    #             )
    #             from safetensors.torch import load_file
    #             pretrained_state_dict = load_file(state_dict_path)
    #             print("Loaded state dict from safetensors")
    #         except:
    #             state_dict_path = hf_hub_download(
    #                 repo_id=vision_tower_cfg.diffusion_name_or_path,
    #                 filename="diffusion_pytorch_model.bin",
    #                 subfolder="transformer"
    #             )
    #             pretrained_state_dict = torch.load(state_dict_path, map_location='cpu')
    #             print("Loaded state dict from .bin file")

    #         # Create model with modified config
            sana = SanaTransformer2DModel.from_config(config_dict, torch_dtype=torch.bfloat16)

    #         # Use direct parameter copying to avoid any loading mechanisms
    #         loaded_count = 0
    #         skipped_count = 0
            
    #         with torch.no_grad():
    #             for name, new_param in sana.named_parameters():
    #                 if name in pretrained_state_dict:
    #                     pretrained_param = pretrained_state_dict[name]
    #                     if new_param.shape != pretrained_param.shape or "proj_out" in name or "patch_embed" in name:
    #                         # Shape mismatch - initialize instead
    #                         if new_param.dim() >= 2:
    #                             nn.init.xavier_uniform_(new_param)
    #                         else:
    #                             nn.init.zeros_(new_param)
    #                         skipped_count += 1
    #                         print(f"Initialized mismatched: {name} {new_param.shape} (was {pretrained_param.shape})")
    #                     else:
    #                         new_param.copy_(pretrained_param)
    #                         loaded_count += 1
    #                 else:
    #                     # Parameter not in pretrained model - initialize
    #                     if new_param.dim() >= 2:
    #                         nn.init.xavier_uniform_(new_param)
    #                     else:
    #                         nn.init.zeros_(new_param)
    #                     skipped_count += 1
    #                     print(f"Initialized missing: {name} {new_param.shape}")

    #         print(f"Directly copied {loaded_count} parameters, initialized {skipped_count} parameters")

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            
    else:
        sana = SanaTransformer2DModel.from_pretrained(
            vision_tower_cfg.diffusion_name_or_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )

    
    return sana

def build_sana_64(vision_tower_cfg, **kwargs):
    pass

def build_vae(vision_tower_cfg, **kwargs):
    vae = AutoencoderDC.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="vae", torch_dtype=torch.bfloat16)
    return vae
