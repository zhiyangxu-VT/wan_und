from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from blip3o.constants import IMAGE_TOKEN_IDX
from blip3o.model.blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM
from blip3o.wan import FlowMatchScheduler, load_wan_ti2v_dit, load_wan_ti2v_vae
from blip3o.wan.wan_runner import wan_model_forward


class blip3oQwenWanConfig(Qwen2_5_VLConfig):
    model_type = "blip3o_qwen_wan"


class blip3oQwenWanModel(blip3oMetaModel, Qwen2_5_VLModel):
    config_class = blip3oQwenWanConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenWanModel, self).__init__(config)


class blip3oQwenForCausalLMWan(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    config_class = blip3oQwenWanConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen_wan"
        self.model = blip3oQwenWanModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.wan_dit = None
        self.wan_vae = None
        self.wan_scheduler = FlowMatchScheduler()
        self.wan_context_projector = nn.Linear(config.hidden_size, 4096, bias=False)
        self.wan_scheduler.set_timesteps(1000, training=True)

        self.post_init()

    def get_model(self):
        return self.model

    def initialize_wan_modules(self, model_path, device, dtype, train_wan=False):
        self.wan_dit = load_wan_ti2v_dit(model_path, device=device, dtype=dtype)
        self.wan_vae = load_wan_ti2v_vae(model_path, device=device, dtype=dtype)
        self.wan_dit.requires_grad_(train_wan)
        self.wan_vae.requires_grad_(False)

    def prepare_inputs_labels_for_video(
        self, input_ids, position_ids, attention_mask, past_key_values, labels
    ):
        if input_ids is None or labels is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        output_indicator = labels != -100
        image_idx = input_ids == IMAGE_TOKEN_IDX
        gen_img_idx = torch.logical_and(output_indicator, image_idx)

        text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        latent_queries = latent_queries.contiguous().view(-1, latent_queries.shape[-1])
        text_embeds = text_embeds.clone()
        text_embeds[gen_img_idx] = latent_queries
        labels[image_idx] = -100
        return None, position_ids, attention_mask, past_key_values, text_embeds, labels

    def _encode_video_latents(self, gen_video):
        videos = [video for video in gen_video]
        device = next(self.wan_vae.parameters()).device
        latents = self.wan_vae.encode(videos, device=device)
        return latents

    def _encode_input_image_latents(self, input_image):
        videos = [image.unsqueeze(1) for image in input_image]
        device = next(self.wan_vae.parameters()).device
        latents = self.wan_vae.encode(videos, device=device)
        return latents

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ids: Optional[list] = None,
        i_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_video: Optional[torch.FloatTensor] = None,
        input_image: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_video(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        total_loss = None
        if labels is not None and gen_video is not None:
            if i_s_pos is None:
                raise ValueError("i_s_pos is required for WAN diffusion training")
            img_hidden_states = []
            for b in range(hidden_states.shape[0]):
                img_hidden_states.append(hidden_states[b, i_s_pos[b]:i_s_pos[b] + 64, :])
            img_hidden_states = torch.stack(img_hidden_states, dim=0)
            context = self.wan_context_projector(img_hidden_states)

            latents = self._encode_video_latents(gen_video)
            if input_image is None:
                input_image = gen_video[:, :, 0]
            input_latents = self._encode_input_image_latents(input_image)
            latents[:, :, 0:1] = input_latents

            noise = torch.randn_like(latents, device=latents.device)
            timestep_id = torch.randint(0, len(self.wan_scheduler.timesteps), (1,))
            timestep = self.wan_scheduler.timesteps[timestep_id].to(device=latents.device, dtype=latents.dtype)
            noisy_latents = self.wan_scheduler.add_noise(latents, noise, timestep)
            target = self.wan_scheduler.training_target(latents, noise, timestep)

            noise_pred = wan_model_forward(
                self.wan_dit,
                noisy_latents,
                timestep,
                context,
                fuse_vae_embedding_in_latents=True,
                use_gradient_checkpointing=self.training,
                use_gradient_checkpointing_offload=False,
            )
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            loss = loss * self.wan_scheduler.training_weight(timestep)
            total_loss = loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
