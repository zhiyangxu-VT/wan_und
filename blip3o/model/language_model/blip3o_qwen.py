from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.integrations import is_wandb_available
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from blip3o.utils import rank0_print

CONCAT_AS_HEIGHT = True

if is_wandb_available():
    import wandb


class blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen"

class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)
        self.config = config

class blip3oQwenForCausalLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None
        config.use_tar_siglip_features = config.use_tar_siglip_features
        self.config = config

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.model.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.model.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        und_images: Optional[List[torch.FloatTensor]] = None,
        und_images_for_vae: Optional[List[torch.FloatTensor]] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, und_images)
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
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
    

        # if labels is not None:
        #     shift_logits = logits[..., :-1, :].contiguous()    # (B, L-1, V)
        #     shift_labels = labels[..., 1:].contiguous()        # (B, L-1)

        #     batch_size, seq_len = shift_labels.size()
        #     mask = torch.zeros_like(shift_labels, dtype=torch.bool)

        #     for b in range(batch_size):
        #         label_row = labels[b]

        #         # find last occurrence
        #         start_mask = (label_row == self.config.image_start_tag_id)
        #         end_mask   = (label_row == self.config.image_end_tag_id)

        #         if start_mask.any() and end_mask.any():
        #             s = (label_row.size(0) - 1) - torch.flip(start_mask, dims=[0]).float().argmax().item()
        #             e = (label_row.size(0) - 1) - torch.flip(end_mask,   dims=[0]).float().argmax().item()

        #             if e > s:
        #                 # shift alignment; inclusive of both s and e
        #                 mask[b, s-1:e] = True

        #     # ignore tokens outside the region
        #     masked_labels = shift_labels.masked_fill(~mask, -100)

        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     masked_labels = masked_labels.view(-1).to(shift_logits.device)

        #     loss = loss_fct(shift_logits, masked_labels)


        if target_images is not None:
            vae = self.model.get_sana_vae()
            latents = vae.encode(target_images).latent
            if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            if self.config.use_und_image_vae_as_noise:
                # assuume und_images_for_vae is a list of list of images
                # need to concat them first to the shape of bs, 3, w, h.
                # need to assert that only one image for one sample
                batch_und_images_for_vae = []
                for und_image_vae in und_images_for_vae:
                    assert len(und_image_vae) == 1, "Only one image for one sample is currently supported"
                    batch_und_images_for_vae.append(und_image_vae[0])
                und_images_for_vae_concat = torch.stack(batch_und_images_for_vae, dim=0)
                ref_latents = vae.encode(und_images_for_vae_concat).latent
                if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                    ref_latents = ref_latents - vae.config.shift_factor
                ref_latents = ref_latents * vae.config.scaling_factor
                noise = torch.randn_like(ref_latents, device=ref_latents.device)
                # noise = torch.cat([noise, ref_latents], dim=1)
            else:
                noise = torch.randn_like(latents, device=latents.device)
            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.model.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.model.noise_scheduler.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            
            if getattr(self.config, "use_und_image_vae_as_noise", False):
                if CONCAT_AS_HEIGHT:
                    noisy_latents = torch.cat([noisy_latents, ref_latents], dim=-1)
                else:
                    noisy_latents = torch.cat([noisy_latents, ref_latents], dim=1)  # Channel dimension
                noisy_latents = noisy_latents.to(torch.bfloat16)
                # noisy_latents = self.model.und_image_vae_as_noise_connector(noisy_latents)
            
            sana = self.model.get_sana()


            start_pos = (labels == self.config.image_start_tag_id).float().argmax(dim=1)   
            end_pos   = (labels == self.config.image_end_tag_id).float().argmax(dim=1)   

            selected_hidden_states = []                       
            for b in range(hidden_states.size(0)):          
                start = start_pos[b].item() + 1         
                end = end_pos[b].item()      
                hidden_states_filter = hidden_states[b, start:end, :]      
                if hidden_states_filter.size(1) != 730:
                    hidden_states_filter = hidden_states[b, -730:, :]
                selected_hidden_states.append(hidden_states_filter) 

            selected_hidden_states = torch.stack(selected_hidden_states, dim=0)
            # selected_hidden_states = []
            # for b in range(hidden_states.size(0)):
            #     label_row = labels[b]

            #     # find *last* start/end
            #     start_mask = (label_row == self.config.image_start_tag_id)
            #     end_mask   = (label_row == self.config.image_end_tag_id)

            #     if start_mask.any() and end_mask.any():
            #         s = (label_row.size(0) - 1) - torch.flip(start_mask, dims=[0]).float().argmax().item()
            #         e = (label_row.size(0) - 1) - torch.flip(end_mask,   dims=[0]).float().argmax().item()

            #         if e > s:
            #             # pick content strictly between <image_start> and <image_end>
            #             hidden_states_filter = hidden_states[b, s+1:e, :]
            #         else:
            #             hidden_states_filter = hidden_states[b, -730:, :]
            #     else:
            #         # fallback if no markers
            #         hidden_states_filter = hidden_states[b, -730:, :]

            #     # pad/trim if not exactly 730
            #     if hidden_states_filter.size(0) != 730:
            #         hidden_states_filter = hidden_states[b, -730:, :]

            #     selected_hidden_states.append(hidden_states_filter)

            # selected_hidden_states = torch.stack(selected_hidden_states, dim=0)
            
            if self.config.use_und_image_vae and not self.config.only_use_und_image_vae_as_noise:
                multimodal_context_condition = self.model.diffusion_connector(selected_hidden_states)
                und_image_vae_condition_list = []
                for und_image_vae in und_images_for_vae:
                    und_image_vae_latents = vae.encode(und_image_vae[0].unsqueeze(0)).latent
                    if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                        und_image_vae_latents = und_image_vae_latents - vae.config.shift_factor
                    und_image_vae_latents = und_image_vae_latents * vae.config.scaling_factor
                    # und_image_vae_latents: [batch, 32, 32, 32], the first 32 is the feature dimension
                    # need to flatten the last two dimensions, then map the first 32 dimension to 2304 dimension using the connector
                    und_image_vae_latents = und_image_vae_latents.movedim(1, -1)
                    und_image_vae_latents = und_image_vae_latents.reshape(-1, und_image_vae_latents.shape[-1])
                    und_iamge_vae_condition = self.model.und_image_vae_connector(und_image_vae_latents)
                    und_image_vae_condition_list.append(und_iamge_vae_condition)
                und_image_vae_condition = torch.stack(und_image_vae_condition_list, dim=0)

                diffusion_condition = torch.cat([multimodal_context_condition, und_image_vae_condition], dim=1)
                diffusion_condition = self.mask_drop(diffusion_condition)
            else:
                diffusion_condition = self.model.diffusion_connector(selected_hidden_states)
                diffusion_condition = self.mask_drop(diffusion_condition)
            
            diffusion_pred = sana(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=diffusion_condition,
                encoder_attention_mask=None,
            ).sample
            
            if getattr(self.config, "use_und_image_vae_as_noise", False):
                if CONCAT_AS_HEIGHT:
                    diffusion_pred = diffusion_pred[..., :latents.shape[-1]]
                else:
                    # Split the channel dimension back to original latent channels
                    diffusion_pred = diffusion_pred[:, :latents.shape[1]]  # Keep only original channels
                

            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            diff_loss = torch.mean(
                (weighting.float() * (diffusion_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            diff_loss = diff_loss.mean()
            loss_ce = loss.detach().clone()
            rank0_print(f" Cross-entropy loss {loss_ce}, Diffusion loss {diff_loss} ")
            loss += diff_loss

        if is_wandb_available() and wandb.run is not None:
          wandb.log({
          "cross_entropy_loss": loss_ce.item(),
          "diffusion_loss": diff_loss.item()
            })



        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("blip3o_qwen", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForCausalLM)
