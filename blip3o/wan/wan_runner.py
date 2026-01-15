import torch
from einops import rearrange

from .wan_video_dit import WanModel, sinusoidal_embedding_1d


def _build_timestep(dit: WanModel, latents: torch.Tensor, timestep: torch.Tensor, fuse_vae_embedding_in_latents: bool):
    timestep = timestep.to(device=latents.device, dtype=latents.dtype)
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        patch_area = dit.patch_size[1] * dit.patch_size[2]
        tokens_per_frame = (latents.shape[3] * latents.shape[4]) // patch_area
        if tokens_per_frame <= 0:
            tokens_per_frame = 1
        first = torch.zeros((1, tokens_per_frame), dtype=latents.dtype, device=latents.device)
        if latents.shape[2] > 1:
            rest = torch.ones((latents.shape[2] - 1, tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep
            timestep_seq = torch.cat([first, rest], dim=0).flatten()
        else:
            timestep_seq = first.flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_seq).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    return t, t_mod


def wan_model_forward(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    t, t_mod = _build_timestep(dit, latents, timestep, fuse_vae_embedding_in_latents)
    context = dit.text_embedding(context)

    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        if use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
