import os
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_VISION_START_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    DEFAULT_IMAGE_PAD_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    VISION_START_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
    IMAGE_PAD_TOKEN_INDEX,
)
from blip3o.utils import rank0_print
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_decoder.builder import build_sana, build_vae
from diffusers.models.normalization import RMSNorm
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel
import math

class blip3oMetaModel:

    def __init__(self, config):
        super(blip3oMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)

            self.sana = build_sana(config)
            self.sana_vae = build_vae(config)
            self.use_tar_siglip_features = getattr(config, "use_tar_siglip_features", False)
            rank0_print(f"Model arch: use_tar_siglip_features = {self.use_tar_siglip_features}")
            norm = RMSNorm(2304, eps=1e-5, elementwise_affine=True)

            with torch.no_grad():
                norm.weight.fill_(math.sqrt(5.5))
            self.diffusion_connector = nn.Sequential(
                nn.Linear(config.hidden_size, 2304),
                nn.GELU(approximate="tanh"),
                nn.Linear(2304, 2304),
                norm,
            )
            if getattr(self.config, "use_tar_siglip_features", False):
                norm_siglip_features_connector = RMSNorm(self.config.hidden_size, eps=1e-5, elementwise_affine=True)
                # with torch.no_grad():
                #     norm_siglip_features_connector.weight.fill_(math.sqrt(5.5))
                self.tar_siglip_features_connector = nn.Sequential(
                    nn.Linear(1152, config.hidden_size),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    norm_siglip_features_connector,
                )
            if getattr(self.config, "use_und_image_vae", False):
                norm_und_image_vae_connector = RMSNorm(2304, eps=1e-5, elementwise_affine=True)
                self.und_image_vae_connector = nn.Sequential(
                    nn.Linear(32, 2304),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(2304, 2304),
                    norm_und_image_vae_connector,
                )
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.diffusion_name_or_path, subfolder="scheduler")
            
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.diffusion_name_or_path, subfolder="scheduler")
            

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


    def get_sana(self):
        sana = getattr(self, 'sana', None)
        if type(sana) is list:
            sana = sana[0]
        if sana is not None:
            sana.to(self.device)
        return sana

    def get_sana_vae(self):
        sana_vae = getattr(self, 'sana_vae', None)
        if type(sana_vae) is list:
            sana_vae = sana_vae[0]
        if sana_vae is not None:
            sana_vae.to(self.device)
        return sana_vae

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()


        if self.get_sana() is None:
            sana = build_sana(model_args)
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_args.diffusion_name_or_path, subfolder="scheduler"
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_args.diffusion_name_or_path, subfolder="scheduler")

            if fsdp is not None and len(fsdp) > 0:
                self.sana = [sana]
            else:
                self.sana = sana
        else:
            if fsdp is not None and len(fsdp) > 0:
                sana = self.sana[0]
            else:
                sana = self.sana


        if self.get_sana_vae() is None:
            sana_vae = build_vae(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.sana_vae = [sana_vae]
            else:
                self.sana_vae = sana_vae
        else:
            if fsdp is not None and len(fsdp) > 0:
                sana_vae = self.sana_vae[0]
            else:
                sana_vae = self.sana_vae


        if getattr(self, 'diffusion_connector', None) is None:
            norm = RMSNorm(2304, eps=1e-5, elementwise_affine=True)
            with torch.no_grad():
                norm.weight.fill_(math.sqrt(5.5))
            self.diffusion_connector = nn.Sequential(
                nn.Linear(self.config.hidden_size, 2304),
                nn.GELU(approximate="tanh"),
                nn.Linear(2304, 2304),
                norm,
            )
        else:
            for p in self.diffusion_connector.parameters():
                p.requires_grad = True
        
        if getattr(self, 'tar_siglip_features_connector', None) is None and self.config.use_tar_siglip_features:
            norm_siglip_features_connector = RMSNorm(self.config.hidden_size, eps=1e-5, elementwise_affine=True)
            
            self.tar_siglip_features_connector = nn.Sequential(
                nn.Linear(1152, self.config.hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                norm_siglip_features_connector,
                )
                
                # Initialize RMSNorm weight properly for bfloat16 stability
                # nn.init.constant_(self.tar_siglip_features_connector[-1].weight, 0.1)
        elif self.config.use_tar_siglip_features:
            # Proper weight initialization to prevent NaN
            with torch.no_grad():
                self.tar_siglip_features_connector[-1].weight.fill_(math.sqrt(5.5))
                nn.init.xavier_uniform_(self.tar_siglip_features_connector[0].weight, gain=0.1)
                nn.init.xavier_uniform_(self.tar_siglip_features_connector[2].weight, gain=0.1)
                self.tar_siglip_features_connector[0].bias.zero_()
                self.tar_siglip_features_connector[2].bias.zero_()
            for p in self.tar_siglip_features_connector.parameters():
                p.requires_grad = True
                
        if getattr(self, 'und_image_vae_connector', None) is None and self.config.use_und_image_vae:
            norm_und_image_vae_connector = RMSNorm(2304, eps=1e-5, elementwise_affine=True)
            self.und_image_vae_connector = nn.Sequential(
                nn.Linear(32, 2304),
                nn.GELU(approximate="tanh"),
                nn.Linear(2304, 2304),
                norm_und_image_vae_connector,
            )
        elif self.config.use_und_image_vae:
            with torch.no_grad():
                self.und_image_vae_connector[-1].weight.fill_(math.sqrt(5.5))
                nn.init.xavier_uniform_(self.und_image_vae_connector[0].weight, gain=0.1)
                nn.init.xavier_uniform_(self.und_image_vae_connector[2].weight, gain=0.1)
                self.und_image_vae_connector[0].bias.zero_()
                self.und_image_vae_connector[2].bias.zero_()
            for p in self.und_image_vae_connector.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


class blip3oMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images, modalities, pool_scale=None):
        image_features = self.get_model().get_vision_tower()(images, pool_scale=pool_scale)

        assert 'tokens' in image_features
        image_tokens = image_features['tokens']

        # discrete features for gen related tasks
        image_tokens = image_tokens + self.config.image_start_token_id
        image_features_embeded = self.get_model().embed_tokens(image_tokens)

        return {'image_features': image_features_embeded, 'image_tokens': image_tokens, 'siglip_features': image_features['image_features']}

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=None, image_sizes=None, und_images=None, tokenizer=None):
        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if not isinstance(modalities, list):
            modalities = [modalities]
        
        # random scale for training, but scale 1 for understanding evaluation
        if self.training:
            # pool_scale = random.choice(vision_tower.pool_scales)
            # fix it for editing training
            pool_scale = 1
        else:
            pool_scale = 1

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images, modalities, pool_scale=pool_scale)
            image_tokens = encoded_image_features['image_tokens']
            encoded_image_features = encoded_image_features['image_features']

            # This is a list, each element is [num_images, patch * patch, dim]
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            if image_tokens is not None:
                image_tokens = torch.split(image_tokens, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                    image_features.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
                if image_tokens is not None:
                    image_tokens = [x.flatten(0, 1) for x in image_tokens]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, modalities, pool_scale=pool_scale)
            image_tokens = image_features['image_tokens']
            image_features = image_features['image_features']
            
        # Handle und_images - compute understanding features for image editing
        und_image_features_per_sample = []
        und_image_tokens_per_sample = []
        if und_images is not None and len(und_images) > 0:
            # und_images is List[List[torch.Tensor]] - batch_size x variable_num_images_per_sample
            for sample_und_images in und_images:
                if len(sample_und_images) > 0:
                    # Concatenate all images for this sample
                    if isinstance(sample_und_images, list):
                        # Multiple images - concatenate them
                        sample_concat = torch.cat([img.unsqueeze(0) for img in sample_und_images], dim=0)
                    else:
                        sample_concat = sample_und_images
                    
                    # Encode the understanding images (no discrete tokens, just features)
                    # sample_features = vision_tower(sample_concat.to(self.device))
                    encoded_image_features = self.encode_images(sample_concat, modalities, pool_scale=pool_scale)
                    und_image_features = encoded_image_features['siglip_features']
                    und_image_tokens = encoded_image_features['image_tokens']
                    
                    und_image_features_per_sample.append(und_image_features)
                    und_image_tokens_per_sample.append(und_image_tokens)
                else:
                    # Empty sample
                    und_image_features_per_sample.append(None)
                    und_image_tokens_per_sample.append(None)

        # Handle understanding images and replace the image pad tokens with the actual understanding images embeddings
        if und_image_features_per_sample:
            # Try to get tokenizer from parameter, stored reference, or config
            tokenizer_to_use = tokenizer or getattr(self, '_tokenizer', None)
            image_pad_token_id = tokenizer_to_use.convert_tokens_to_ids(DEFAULT_IMAGE_PAD_TOKEN)
            
            siglip_features_projected_embeddings = []
            
            for batch_idx, cur_input_ids in enumerate(input_ids):
                if batch_idx < len(und_image_features_per_sample) and und_image_features_per_sample[batch_idx] is not None:
                    # Find image pad token positions
                    image_pad_indices = torch.where(cur_input_ids == image_pad_token_id)[0]
                    if len(image_pad_indices) > 0:
                        # Get the understanding images features and tokens
                        und_image_features = und_image_features_per_sample[batch_idx]
                        und_image_tokens = und_image_tokens_per_sample[batch_idx]
                        
                        assert und_image_tokens is not None or und_image_features is not None, "und_image_tokens and und_image_features are both None"
                        
                        # if self.get_model().use_tar_siglip_features:
                        #     # use features instead of tokens here and the rest of the code is the same
                        #     und_image_tokens = self.get_model().tar_siglip_features_connector(und_image_features)
                        # Flatten tokens if needed
                        if und_image_tokens.dim() > 1:
                            und_image_tokens = und_image_tokens.flatten()
                        
                        # Replace pad tokens with understanding tokens (limit to available tokens)
                        num_replacements = min(len(image_pad_indices), len(und_image_tokens))
                        if num_replacements > 0:
                            # if use_tar_siglip_features is True, we don't replace the image tokens for now, will replace it later after get the embeddings
                            if not self.config.use_tar_siglip_features:
                                cur_input_ids[image_pad_indices[:num_replacements]] = und_image_tokens[:num_replacements]
                            else:
                                # keep the last dimension and flatten the rest, don't assume the shape of the features
                                siglip_features_projected_embeddings.append(self.get_model().tar_siglip_features_connector(und_image_features.reshape(-1, und_image_features.shape[-1])))
                            
                            # Update attention mask and labels for replaced positions
                            if attention_mask is not None:
                                attention_mask[batch_idx, image_pad_indices[:num_replacements]] = True
                            if labels is not None:
                                labels[batch_idx, image_pad_indices[:num_replacements]] = IGNORE_INDEX
                else:
                    siglip_features_projected_embeddings.append(None)
                
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                # cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_input_embeds_1[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            
            if self.config.use_tar_siglip_features and und_images[batch_idx]:
                siglip_features_projected_embedding = siglip_features_projected_embeddings[batch_idx]
                # do the replacement here, replace the image tokens with the siglip features projected embeddings
                # use the image_pad_indices to replace the image tokens
                image_pad_indices = torch.where(cur_input_ids == image_pad_token_id)[0]
                cur_input_embeds[image_pad_indices] = siglip_features_projected_embedding
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        rank0_print("Error image_features[cur_image_idx]!")
                        break
                    # [Assisant\n<start_image><image><end_image>]
                    if self.config.image_start_tag_id == cur_labels_noim[i][-1] and image_tokens is not None:
                        cur_image_tokens = image_tokens[cur_image_idx]
                        if pool_scale is not None:
                            pool_token = self.config.scale_start_token_id + pool_scale - 1
                            pool_token = torch.tensor([pool_token], dtype=torch.long, device=cur_image_tokens.device)
                            cur_image_tokens = torch.cat([pool_token, cur_image_tokens])
                            pool_embed = self.get_model().embed_tokens(pool_token)
                            cur_image_features = torch.cat([pool_embed, cur_image_features])
                    else:
                        cur_image_tokens = torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(cur_image_tokens)
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # Store tokenizer reference for later use
        self._tokenizer = tokenizer
        
        total_num_new_tokens = 0
        vocab_size = len(tokenizer)
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.config.image_start_tag_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            self.config.image_end_tag_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            total_num_new_tokens += num_new_tokens
            self.resize_token_embeddings(vocab_size + total_num_new_tokens)

        # Add vision understanding tokens
        # vision_tokens = [DEFAULT_VISION_START_TOKEN, DEFAULT_VISION_END_TOKEN, DEFAULT_IMAGE_PAD_TOKEN]
        # num_vision_tokens = tokenizer.add_tokens(vision_tokens, special_tokens=True)
        # self.config.vision_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_VISION_START_TOKEN)
        # self.config.vision_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN)
        # self.config.image_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PAD_TOKEN)
        # total_num_new_tokens += num_vision_tokens
        # self.resize_token_embeddings(vocab_size + total_num_new_tokens)

        if model_args.num_scale_tokens > 0:
            scale_tokens = [model_args.scale_token_format.format(str(i)) for i in range(model_args.num_scale_tokens)]
            num_new_tokens = tokenizer.add_tokens(scale_tokens, special_tokens=False)
            self.config.scale_start_token_id = tokenizer.convert_tokens_to_ids(scale_tokens[0])
            self.config.scale_end_token_id = tokenizer.convert_tokens_to_ids(scale_tokens[-1])
            self.config.num_scale_tokens = model_args.num_scale_tokens
            total_num_new_tokens += num_new_tokens
            self.resize_token_embeddings(vocab_size + total_num_new_tokens)

        if model_args.num_image_tokens > 0:
            image_tokens = [model_args.image_token_format.format(str(i)) for i in range(model_args.num_image_tokens)]
            num_new_tokens = tokenizer.add_tokens(image_tokens, special_tokens=False)
            self.config.image_start_token_id = tokenizer.convert_tokens_to_ids(image_tokens[0])
            self.config.image_end_token_id = tokenizer.convert_tokens_to_ids(image_tokens[-1])
            self.config.num_image_tokens = model_args.num_image_tokens

            total_num_new_tokens += num_new_tokens
            self.resize_token_embeddings(vocab_size + total_num_new_tokens)
        if num_new_tokens > 0:
            self.config.num_new_tokens = num_new_tokens
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
            vision_tower = self.get_vision_tower()
            if model_args.load_embeddings_from_vision and vision_tower is not None:                
                vision_embeddings = vision_tower.get_embedding()
                if model_args.num_image_tokens == vision_embeddings.shape[0] and input_embeddings.shape[1] == vision_embeddings.shape[1]:
                    rank0_print("Load vision embeddings from vision tower.")
                    input_embeddings[self.config.image_start_token_id:self.config.image_end_token_id+1] = vision_embeddings