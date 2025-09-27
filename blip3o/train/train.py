import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import deepspeed
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from blip3o.data import make_supervised_data_module
from blip3o.model import blip3oQwenForCausalLM
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o.utils import rank0_print
from tabulate import tabulate

torch.multiprocessing.set_sharing_strategy("file_system")

local_rank = None

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    diffusion_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from blip3oLlama, blip3oMixtral, blip3oMistral, Llama"})
    mm_tunable_parts: Optional[str] = field(default="mm_language_model")
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    delay_load: Optional[bool] = field(default=True)
    num_image_tokens: Optional[int] = field(default=-1)
    image_token_format: str = field(default="<I{}>")
    num_scale_tokens: Optional[int] = field(default=3)
    scale_token_format: str = field(default="<S{}>")
    load_embeddings_from_vision: Optional[bool] = field(default=False)
    use_tar_siglip_features: Optional[bool] = field(default=False)
    use_und_image_vae: Optional[bool] = field(default=False)
    use_und_image_vae_as_noise: Optional[bool] = field(default=False)
    only_use_und_image_vae_as_noise: Optional[bool] = field(default=False)
@dataclass
class DataArguments:
    data_list: str = field(default=None, metadata={"help": "Comma-separated list of dataset paths."})
    data_list_weights: str = field(default=None, metadata={"help": "Comma-separated list of dataset weights."})
    subsample_ratio: float = field(default=1.0, metadata={"help": "Subsample ratio for the entire training."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    dataset_cls: str = field(default="blip3o")

    def __post_init__(self):
        if self.data_list is not None:
            self.data_list = [path.strip() for path in self.data_list.split(',')]
            if self.data_list_weights is not None:
                assert len(self.data_list) == len(self.data_list_weights), "The number of datasets and weights must be the same"
                self.data_list_weights = [float(weight.strip()) for weight in self.data_list_weights.split(',')]
            else:
                self.data_list_weights = [1.0] * len(self.data_list)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    dispatch_batches: Optional[bool] = field(default=None)
    split_batches: Optional[bool] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    
    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_model(model_args, training_args):
    customized_kwargs = {}
    overwrite_config = {}

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    # Set use_tar_siglip_features in config before model initialization
    overwrite_config["use_tar_siglip_features"] = model_args.use_tar_siglip_features

    overwrite_config["use_und_image_vae"] = model_args.use_und_image_vae
    overwrite_config["use_und_image_vae_as_noise"] = model_args.use_und_image_vae_as_noise
    overwrite_config["only_use_und_image_vae_as_noise"] = model_args.only_use_und_image_vae_as_noise
    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    model = blip3oQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs)
    
    # reinitialize the sana part, change the in_channel and out_channel to 64, and reinitialize the sana part
    from blip3o.model.language_model.blip3o_qwen import CONCAT_AS_HEIGHT
    if model_args.use_und_image_vae_as_noise and not CONCAT_AS_HEIGHT:
        rank0_print(f"Reinitializing Sana with 64 channels...")
        sana = model.get_model().get_sana()
        config_new_sana = dict(sana.config)
        config_new_sana['in_channels'] = 64
        config_new_sana['out_channels'] = 64
        # rebuild the sana part with the new config and load the matched old weights
        from diffusers import SanaTransformer2DModel
        sana_new = SanaTransformer2DModel.from_config(config_new_sana, torch_dtype=torch.bfloat16)
        # filter out the shape-mismatched parameters and the proj_out and patch_embed parameters
        filtered_state_dict = {
            name: param for name, param in sana.state_dict().items()
            if not ("proj_out" in name or "patch_embed" in name)
        }
        load_result = sana_new.load_state_dict(filtered_state_dict, strict=False)
        rank0_print(f"Missing keys:")
        rank0_print(load_result.missing_keys)
        rank0_print(f"Unexpected keys:")
        rank0_print(load_result.unexpected_keys)
        model.get_model().sana = sana_new
    
    return model


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Debug: Print the parsed arguments
    rank0_print(f"Parsed use_tar_siglip_features: {model_args.use_tar_siglip_features}")

    local_rank = training_args.local_rank

    model = get_model(model_args, training_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.diffusion_name_or_path = model_args.diffusion_name_or_path

        
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        rank0_print(f"Model config use_tar_siglip_features: {model.config.use_tar_siglip_features}")
        rank0_print(f"Model config use_und_image_vae: {model.config.use_und_image_vae}")
        rank0_print(f"Model config use_und_image_vae_as_noise: {model.config.use_und_image_vae_as_noise}")
        rank0_print(f"Model config only_use_und_image_vae_as_noise: {model.config.only_use_und_image_vae_as_noise}")
        training_args.use_im_start_end = model_args.mm_use_im_start_end

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        ### Deciding train which part of the model
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        vision_tower.eval()
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name:
                    param.requires_grad_(True)
        if 'mm_embedding' in tunable_parts:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or 'lm_head' in name:
                    param.requires_grad_(True)

        ## freeze sana except the caption projection
        for name, param in model.named_parameters():
            if "sana" in name:
                param.requires_grad_(False)

        for name, param in model.named_parameters():
            if "caption" in name:
                param.requires_grad_(True)
        
        if model.config.use_und_image_vae_as_noise:
            for name, param in model.named_parameters():
                if "sana" in name and ("sana.patch_embed" in name or "sana.proj_out" in name):
                    param.requires_grad_(True)
            
        # if True:
        #     for name, param in model.named_parameters():
        #         if "sana.transformer_blocks" in name:
        #             param.requires_grad_(True)
        
        # # Unfreeze SANA input layer if using concatenated VAE noise mode
        # # This layer has mismatched dimensions and was randomly initialized
        # if model.config.use_und_image_vae_as_noise:
        #     unfrozen_layers = []
        #     for name, param in model.named_parameters():
        #         if "sana" in name and ("pos_embed" in name or "x_embedder" in name or "conv_in" in name or "patch_embed" in name):
        #             param.requires_grad_(True)
        #             unfrozen_layers.append(name)
        #     if unfrozen_layers:
        #         rank0_print(f"🔓 Unfrozen SANA input layers for concatenated VAE mode: {unfrozen_layers}")
        #     else:
        #         rank0_print("⚠️  Warning: No SANA input layers found to unfreeze for concatenated VAE mode")   
                

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        for name, p in model.named_parameters():
            if p.requires_grad:
                rank0_print(f"Trainable parameter: {name}")
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = blip3oTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)


    if trainer.is_world_process_zero():
        # Debug: Comment out parameter listing for now
        # stat = []
        # for i, (n, p) in enumerate(trainer.model.named_parameters()):
        #     stat.append([i, n, p.shape, p.requires_grad])
        # print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        pass

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
