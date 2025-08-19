import copy
import glob
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
import glob
from datasets import load_from_disk, Sequence, Value
from datasets import Image as DatasetImage
import datasets
from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from blip3o.utils import rank0_print


ImageFile.LOAD_TRUNCATED_IMAGES = True


X2I2_BASE_IMAGE_DIRS = {
    # "X2I_x2i2_reflect": "/fsx/home/lxue/repos/datasets/X2I2/images/reflect/reflect",
    "X2I_x2i2_inpaint": "/fsx/home/lxue/repos/datasets/X2I2_latest/X2I2/images/inpaint_edit",
    "X2I_x2i2_video_edit": "/fsx/home/lxue/repos/datasets/X2I2/images/video_edit",
    "X2I_x2i2_video_icedit": "/fsx/home/lxue/repos/datasets/X2I2/images/video_icedit/edit_ip",
    "X2I_x2i2_video_icgen": "/fsx/home/lxue/repos/datasets/X2I2/images/video_icgen",
    "X2I_x2i2_video_interleave": "/fsx/home/lxue/repos/datasets/X2I2/images/video_interleave/x_mv",
    "X2I_sharegpt4o": "/fsx/home/lxue/repos/datasets/sharegpt4o_all/x2i",
    # "X2I_sharegpt4o_t2i": "/fsx/home/lxue/repos/datasets/sharegpt4o_t2i/sharegpt4o",
}

## target transform for sana
target_transform = v2.Compose(
    [
        v2.Resize(1024),
        v2.CenterCrop(1024),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_multimodal(sources, data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources

def preprocess_multimodal_x2i(sources, data_args) -> Dict:
    # For image editing, we need custom handling since it has both und and gen tokens
    sources_copy_conv = copy.deepcopy([sources["conversations"]])
    n_und_query = data_args.n_und_query  # Per image query count
    input_images = sources["input_images"]
    num_input_images = len(input_images)
    total_und_tokens = num_input_images * n_und_query
    # Replace <image> tokens in human message with understanding placeholders
    # and remove <image> token from gpt message
    und_placeholder = "<|vision_start|>" + "<|image_pad|>" * total_und_tokens + "<|vision_end|>"
    for source in sources_copy_conv:
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return sources_copy_conv

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    #tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])


        # target += [IGNORE_INDEX] * len(input_id)
        target += input_id

        for conv in source:
            # Make sure blip3o data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                # target += [IGNORE_INDEX] * len(encode_id)
                target += encode_id

            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  
        labels=targets,  
    )



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_list: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        self.data_args.n_und_query = 729
        list_data_dict = []

        # assume the dataset is a list of paths
        for dataset_path in data_list:
            if 'metaquery' in dataset_path.lower():

                train_dataset = load_from_disk(dataset_path, keep_in_memory=False)
                
            elif 'x2i2' in dataset_path.lower():
                train_dataset = load_dataset("json", data_files=dataset_path, split="train", num_proc=1)
                if "reflect" in dataset_path.lower():
                    def map_reflect(batch):
                        txt = []
                        for i, col in enumerate(batch["used_instruction"]):
                            if col in batch and i < len(batch[col]):
                                txt.append(batch[col][i])
                            elif "instruction" in batch and i < len(batch["instruction"]):
                                txt.append(batch["instruction"][i])
                            else:
                                txt.append("N/A")  # fallback placeholder
                        return {
                            'type': ['X2I_x2i2_reflect'] * len(batch["image_path"]),
                            'image': [None] * len(batch["image_path"]),
                            'txt': txt,
                            'input_images': [None] * len(batch["image_path"])
                        }
                    train_dataset = train_dataset.rename_column("image", "image_path")
                    train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
                    train_dataset = train_dataset.map(map_reflect, batched=True, batch_size=2000, num_proc=1)
                    print("loaded reflect dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                elif "inpaint" in dataset_path.lower():
                    train_dataset = train_dataset.rename_column("output_image", "image_path")
                    train_dataset = train_dataset.rename_column("instruction", "txt")
                    train_dataset = train_dataset.add_column('type', len(train_dataset) * ['X2I_x2i2_inpaint'])
                    train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
                    train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
                    train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
                    print("loaded inpaint dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                elif "video_edit" in dataset_path.lower():
                    # train_dataset = train_dataset.add_column('type', len(train_dataset) * ['X2I_x2i2_video_edit'])
                    # rename the colume of output_image to image and the instruction to txt
                    train_dataset = train_dataset.rename_column("output_image", "image_path")
                    train_dataset = train_dataset.rename_column("instruction", "txt")
                    train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
                    def map_video_edit(batch):
                        return {
                            'type': ['X2I_x2i2_video_edit'] * len(batch["image_path"]),
                            'image': [None] * len(batch["image_path"]),
                            'input_images': [[]] * len(batch["image_path"])
                        }
                    train_dataset = train_dataset.map(map_video_edit, batched=True, batch_size=2000, num_proc=1)
                    print("loaded video edit dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                elif "video_icedit" in dataset_path.lower():
                    train_dataset = train_dataset.rename_column("instruction", "txt")
                    train_dataset = train_dataset.rename_column("output_image", "image_path")
                    train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
                    def map_video_icedit(batch):
                        return {
                            'type': ['X2I_x2i2_video_icedit'] * len(batch["image_path"]),
                            'image': [None] * len(batch["image_path"]),
                            'input_images': [[]] * len(batch["image_path"])
                        }
                    train_dataset = train_dataset.map(map_video_icedit, batched=True, batch_size=2000, num_proc=1)
                    print("loaded video icedit dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                # icgen has both long and short versions
                elif "video_icgen" in dataset_path.lower():
                    train_dataset = train_dataset.rename_column("instruction", "txt")
                    train_dataset = train_dataset.rename_column("output_image", "image_path")
                    train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
                    def map_video_icgen(batch):
                        return {
                            'type': ['X2I_x2i2_video_icgen'] * len(batch["image_path"]),
                            'image': [None] * len(batch["image_path"]),
                            'input_images': [[]] * len(batch["image_path"])
                        }
                    train_dataset = train_dataset.map(map_video_icgen, batched=True, batch_size=2000, num_proc=1)
                    print("loaded video icgen dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                #TODO: interleaved data instruction contains the image token, needs some more processing
                elif "video_interleave" in dataset_path.lower():
                    train_dataset = train_dataset.add_column('type', len(train_dataset) * ['X2I_x2i2_video_interleave'])
                else:
                    raise ValueError(f"No X2I2 dataset found in {dataset_path}")
                

            elif 'BLIP3o-60k' in dataset_path:
                # BLIP3o-60k contains TAR files, load them as webdataset
                tar_files = sorted(glob.glob(f"{dataset_path}/*.tar"))
                if tar_files:
                    train_dataset = load_dataset("webdataset", data_files=tar_files, split="train", num_proc=1, cache_dir='/fsx/home/lxue/repos/BLIP3o/edit_data')
                    train_dataset = train_dataset.rename_column("jpg", "image")
                    train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
                    train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
                    train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
                    train_dataset = train_dataset.add_column('input_images_paths', len(train_dataset) * [None])
                    # repeat this dataset 20 times
                    # train_dataset = train_dataset.repeat(10)
                    print("loaded BLIP3o-60k dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                else:
                    raise ValueError(f"No TAR files found in {dataset_path}")
            elif 'sharegpt4o' in dataset_path.lower():
                if 't2i' in dataset_path.lower():
                    # load the tar files in the dataset_path
                    tar_files = sorted(glob.glob(f"{dataset_path}/*.tar"))
                    if len(tar_files) == 0:
                        raise ValueError(f"No TAR files found in {dataset_path}")
                    train_dataset = load_dataset("webdataset", data_files=tar_files, split="train", num_proc=1, cache_dir='/fsx/home/lxue/repos/BLIP3o/edit_data/sharegpt4o_t2i')
                    train_dataset = train_dataset.rename_column("jpg", "image")
                    train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
                    train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
                    train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
                    train_dataset = train_dataset.add_column('input_images_paths', len(train_dataset) * [None])
                    # repeat this dataset 10 times
                    # train_dataset = train_dataset.repeat(10)
                    print("loaded sharegpt4o t2i dataset with ", len(train_dataset), " samples", "from ", dataset_path)
                elif 'x2i' in dataset_path.lower():
                    # train_dataset = load_dataset("webdataset", data_files=tar_files, split="train", num_proc=1, cache_dir='/fsx/home/lxue/repos/BLIP3o/edit_data/sharegpt4o_x2i')
                    train_dataset = load_dataset("json", data_files=dataset_path, split="train", num_proc=1)
                    train_dataset = train_dataset.rename_column("output_image", "image_path")
                    train_dataset = train_dataset.rename_column("input_image", "input_images_paths")
                    train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
                    train_dataset = train_dataset.rename_column("input_prompt", "txt")
                    train_dataset = train_dataset.add_column('type', len(train_dataset) * ['X2I_sharegpt4o'])
                    train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
                    # repeat this dataset 10 times
                    # train_dataset = train_dataset.repeat(10)
                    print("loaded sharegpt4o x2i dataset with ", len(train_dataset), " samples", "from ", dataset_path)
            else:
                # assume the dataset is a glob of tar files, like the ones used in the original blip3o
                data_files = sorted(glob.glob(dataset_path))
                train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128, cache_dir='/fsx/sfr/data/lxue/hf_cache')
                train_dataset = train_dataset.rename_column("jpg", "image")
                train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
                train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
                train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
                train_dataset = train_dataset.add_column('input_images_paths', len(train_dataset) * [None])
            
            train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
                ["image", "txt", "type", "image_path", "input_images", "input_images_paths"])])


            print(f"finish loading image {len(train_dataset)}")
            list_data_dict.append(train_dataset)

        print(f"Total samples: {len(train_dataset)}")
        print(f"Features: {train_dataset.features}")

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)
        # random sample 2000000 samples
        # list_data_dict = list_data_dict.select(range(2000000))

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.modality = torch.tensor(0) # 0 is for und task, 1 is for gen task


    def __len__(self):
        return len(self.list_data_dict)


    def process_image(self, image):
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality


    def process_target_image(self, image):
        image = target_transform(image)
        return image


    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = copy.deepcopy(self.list_data_dict[i])
            if not sources["input_images"]:
                sources["input_images"] = sources["input_images_paths"]
            if sources["image"] is None:
                sources["image"] = sources["image_path"]
            
            if not sources["input_images"]:
                sources["input_images"] = []
            if len(sources["input_images"]) > 4:
                sources["input_images"] = sources["input_images"][:4]

            if sources["type"] == "T2I":

                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]

            elif sources["type"] == "I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            
            elif "X2I" in sources["type"]:
                # Check if there are multiple input images
                num_input_images = len(sources.get('input_images', []))
                
                if num_input_images > 1:
                    image_tokens = "<image>"
                    prompt_text = f"Please edit these images according to the following instruction: {sources['txt']}.\n{image_tokens}"
                else:
                    prompt_text = f"Please edit this image according to the following instruction: {sources['txt']}.\n<image>"
                
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": prompt_text
                    },
                    {"from": "gpt", "value": "<image>"},
                ]
    
            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:
                images = []                    
                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = sources["image"]
                    if not isinstance(image_files, list):
                        image_files = [image_files]
                    
                    for img in image_files:
                        try:
                            img = img.convert("RGB")
                            images.append(img)
                        except Exception as e:
                            print(f"Error opening image {img}: {e}")
                            images = None
                            i = random.randint(0, len(self.list_data_dict) - 1)
                            continue
                    
                elif "X2I" in sources["type"]:
                    input_images = sources["input_images"]
                    target_image = sources["image"]
                    try:
                        input_images_pil = []
                        for image in input_images:
                            if isinstance(image, str):
                                image = Image.open(os.path.join(X2I2_BASE_IMAGE_DIRS[sources["type"]], image)).convert("RGB")
                                images.append(image)
                            input_images_pil.append(image)
                        sources["input_images"] = input_images_pil
                        if isinstance(target_image, str):
                            target_image_pil = Image.open(os.path.join(X2I2_BASE_IMAGE_DIRS[sources["type"]], target_image)).convert("RGB")
                        sources["image"] = target_image_pil
                        images.append(target_image_pil)
                    except Exception as e:
                        print(f"Error opening target image {target_image}: {e}")
                        images = None
                        i = random.randint(0, len(self.list_data_dict) - 1)
                        continue


                ## test if can apply img_process 
                if not images is None:
                    try:
                        process_images = [self.process_image(f) for f in images]
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None

                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue
                
                # Handle image editing separately from preprocess_multimodal
                if "X2I" in sources["type"]:
                    sources = preprocess_multimodal_x2i(sources, self.data_args)
                else:
                    sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])

            data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "image" in self.list_data_dict[i]:
                data_dict["image"] = process_images[-1]
                data_dict["target_image"] = self.process_target_image(images[-1])
                data_dict["und_images"] = [image[0] for image in process_images[:-1]]

            data_dict["ids"] = sources["id"] if "id" in sources else "unk"
            return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = []
            batch["modalities"] = []
            batch["images"] = []
            for image_tuple in images:
                batch["image_sizes"].append(image_tuple[1])
                batch["modalities"].append(image_tuple[2])
                batch["images"].append(image_tuple[0])

            target_images = [instance["target_image"] for instance in instances]
            target_images = torch.stack(target_images, dim=0) if target_images else None
            batch["target_images"] = target_images

        batch_und_images = []
        for instance in instances:
            if "und_images" in instance:
                batch_und_images.append(instance["und_images"])
        if len(batch_und_images) > 0:
            batch["und_images"] = batch_und_images
        else:
            batch["und_images"] = None

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        return batch

def get_dataset_cls(name):

    if name == 'mix':
        dataset_cls = LazySupervisedMixDataset
    else:
        raise ValueError(f'Unknown dataset class {name}')
    return dataset_cls

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = get_dataset_cls(data_args.dataset_cls)
    train_dataset = dataset_cls(tokenizer=tokenizer, data_list=data_args.data_list, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)