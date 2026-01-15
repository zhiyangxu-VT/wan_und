import glob
import json
import os
from typing import Dict, Iterable, Optional

import torch

from .wan_video_dit import WanModel
from .wan_video_vae import WanVideoVAE38, WanVideoVAEStateDictConverter


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load .safetensors checkpoints. "
            "Please install it or use a .pth checkpoint."
        ) from exc
    return load_file(path, device="cpu")


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return _load_safetensors(path)
    return torch.load(path, map_location="cpu")


def _load_sharded_safetensors(model_path: str, base_name: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(model_path, f"{base_name}.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        state_dict: Dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(model_path, shard_name)
            state_dict.update(_load_safetensors(shard_path))
        return state_dict

    shards = sorted(glob.glob(os.path.join(model_path, f"{base_name}*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No {base_name}*.safetensors found under {model_path}")
    state_dict: Dict[str, torch.Tensor] = {}
    for shard_path in shards:
        state_dict.update(_load_safetensors(shard_path))
    return state_dict


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _unwrap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in ("state_dict", "model_state", "model"):
        if key in state_dict and isinstance(state_dict[key], dict):
            return state_dict[key]
    return state_dict


def _maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _looks_like_diffusers(state_dict: Dict[str, torch.Tensor]) -> bool:
    sample_keys = list(state_dict.keys())[:50]
    return any("attn1" in k or "condition_embedder" in k for k in sample_keys)


def WanVideoDiTFromDiffusers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    state_dict_ = {}
    for name, value in state_dict.items():
        if name in rename_dict:
            state_dict_[rename_dict[name]] = value
        else:
            name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if name_ in rename_dict:
                name_ = rename_dict[name_]
                name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                state_dict_[name_] = value
    return state_dict_


def WanVideoDiTStateDictConverter(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state_dict_ = {}
    for name, value in state_dict.items():
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model."):]
        state_dict_[name_] = value
    return state_dict_


def _load_wan_dit_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    base_name = "diffusion_pytorch_model"
    candidate = _first_existing([
        os.path.join(model_path, f"{base_name}.safetensors"),
        os.path.join(model_path, f"{base_name}.bin"),
        os.path.join(model_path, f"{base_name}.pth"),
    ])
    if candidate:
        return _unwrap_state_dict(_load_state_dict(candidate))
    return _unwrap_state_dict(_load_sharded_safetensors(model_path, base_name))


def _load_wan_vae_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    candidates = [
        os.path.join(model_path, "Wan2.2_VAE.pth"),
        os.path.join(model_path, "Wan2.2_VAE.safetensors"),
        os.path.join(model_path, "Wan2.2_VAE.bin"),
    ]
    candidate = _first_existing(candidates)
    if candidate is None:
        matches = glob.glob(os.path.join(model_path, "Wan2.2_VAE*"))
        candidate = matches[0] if matches else None
    if candidate is None:
        raise FileNotFoundError(f"Could not find Wan2.2 VAE checkpoint under {model_path}")
    return _unwrap_state_dict(_load_state_dict(candidate))


def load_wan_ti2v_dit(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16) -> WanModel:
    config = dict(
        dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        num_layers=30,
        has_image_input=False,
        has_image_pos_emb=False,
        has_ref_conv=False,
        add_control_adapter=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    model = WanModel(**config)
    state_dict = _load_wan_dit_state_dict(model_path)
    state_dict = _maybe_strip_prefix(state_dict, "model.")
    if _looks_like_diffusers(state_dict):
        state_dict = WanVideoDiTFromDiffusers(state_dict)
    state_dict = WanVideoDiTStateDictConverter(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WAN] Missing DiT keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WAN] Unexpected DiT keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    model.to(device=device, dtype=dtype)
    return model


def load_wan_ti2v_vae(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16) -> WanVideoVAE38:
    model = WanVideoVAE38()
    state_dict = _load_wan_vae_state_dict(model_path)
    converter = WanVideoVAEStateDictConverter()
    if "model_state" in state_dict or not any(k.startswith("model.") for k in state_dict):
        state_dict = converter.from_civitai(state_dict)
    state_dict = _maybe_strip_prefix(state_dict, "model.")
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WAN] Missing VAE keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WAN] Unexpected VAE keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    model.to(device=device, dtype=dtype)
    return model
