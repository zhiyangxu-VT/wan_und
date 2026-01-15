from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_projector.builder import build_down_projector

from blip3o.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_IDX, UND_IMAGE_TOKEN_IDX



class blip3oMetaModel:

    def __init__(self, config):
        super(blip3oMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # self.vision_tower = build_vision_tower(config, delay_load=True)
            # self.mm_projector = build_vision_projector(config)
            self.down_projector = build_down_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


        self.latent_queries = None
        if hasattr(config, "n_query"):
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            print(f" latent query size {self.latent_queries.shape}")


    # def get_vision_tower(self):
    #     vision_tower = getattr(self, 'vision_tower', None)
    #     if type(vision_tower) is list:
    #         vision_tower = vision_tower[0]
    #     return vision_tower


    def initialize_vision_modules(self, model_args, fsdp=None):
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')

        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.n_query = model_args.n_query

        # if getattr(self, 'mm_projector', None) is None:
        #     print("random initiation the mm_project !!!")
        #     self.mm_projector = build_vision_projector(self.config)

        #     if 'unpad' in mm_patch_merge_type:
        #         embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #         self.image_newline = nn.Parameter(
        #             torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
        #         )
        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_projector.parameters():
        #         p.requires_grad = True



        if getattr(self, 'down_projector', None) is None:
            self.down_projector = build_down_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.down_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True


        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class blip3oMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_projector(self):
        return self.get_model().mm_projector
    

    def get_n_query(self):
        return self.get_model().config.n_query

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, und_images, grid_thw, i_s_pos, image_sizes=None
    ):
        vision_tower = self.visual
        if und_images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None


        und_image_embeds = vision_tower(und_images, grid_thw=grid_thw)
            # _, c = und_image_embeds.shape
            # batch_size = und_images.shape[0]
            # und_image_embeds = und_image_embeds.view(batch_size, -1, c)
            # und_image_embeds = und_image_embeds.contiguous().view(-1, c)
            # und_image_embeds = mm_projector(und_image_embeds)

        # else:
        #     num_img = input_ids.shape[0]
        #     dummy = torch.zeros(num_img, 3, 384, 384 , dtype=gen_images.dtype, device=gen_images.device)  # clip (3, 336, 336) 
        #     temp = vision_tower(dummy)
        #     if 'early' in self.get_gen_pooling():
        #         temp = temp[:,:64,:]
        #     num_img, _, c = temp.shape
        #     temp = temp.contiguous().view(-1, c)
        #     temp = mm_projector(temp) * 1e-20
        #     latent_queries += temp




        
        und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        # img_indicator = torch.clone(image_idx)
        input_indicator = labels == -100
        # img_loss_indicator = torch.logical_and(output_indicator, image_idx)
        # img_loss_indicator = torch.cat(
        #     [img_loss_indicator[:, 1:], img_loss_indicator[:, :1]], dim=1)
        
        # img_indicator = torch.cat(
        #     [img_indicator[:, 1:], img_indicator[:, :1]], dim=1)
        
        # if not target_image_embeds is None:
        #     target_image_embeds = target_image_embeds[-img_loss_indicator.sum():,:]
        text_embeds = self.get_model().embed_tokens(input_ids)
        text_embeds = text_embeds.clone()

        und_img_idx = torch.logical_and(input_indicator, und_image_idx)
     

        text_embeds[und_img_idx] = und_image_embeds.to(text_embeds.device)[:und_img_idx.sum(), :]

        labels[image_idx] = -100


        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, None



    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
