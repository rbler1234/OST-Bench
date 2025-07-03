import argparse
import json
import math
import os
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

down_sample_ratio = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

generation_config = dict(max_new_tokens=1024, do_sample=False)


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()

    num_layers = {
        'InternVL2_5-1B': 24,
        'InternVL2_5-2B': 24,
        'InternVL2_5-4B': 36,
        'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48,
        'InternVL2_5-38B': 64,
        'InternVL2_5-78B': 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=12,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((
        image.size[0] // down_sample_ratio,
        image.size[1] // down_sample_ratio,
    ))
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image,
                                image_size=input_size,
                                use_thumbnail=True,
                                max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def prepare_images_for_input(image_paths):
    pixel_values_list = []
    for image_path in image_paths:
        pixel_values_list.append(
            load_image(image_path, max_num=4).to(torch.bfloat16).cuda())
    pixel_values = torch.cat(pixel_values_list, dim=0)
    num_patches_list = [
        pixel_value.size(0) for pixel_value in pixel_values_list
    ]
    return pixel_values, num_patches_list


def internvl_multi_turn(model, user_messages, tokenizer, generation_config):
    results = []
    history = None
    all_image_paths = []
    for user_message in user_messages:
        user_text = user_message[-1]
        image_paths = [image_path for image_path in user_message[:-1]]
        all_image_paths.extend(image_paths)
        pixel_values, num_patches_list = prepare_images_for_input(
            all_image_paths)
        question = user_text
        response, history = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=history,
            return_history=True,
        )
        results.append(response)
    return results


def process_data(data, model, tokenizer, image_root, save_root):
    """Processes multi-turn dialogue data by generating model responses and
    saving enhanced results to JSON.

    This function:
        (1) Processes image paths by joining with the provided root directory
        (2) Generates responses for each dialogue turn using specified model
        (3) Maintains conversation history across turns
        (4) Saves the enhanced data (original inputs + model responses) to JSON
        (5) Includes timing metrics for performance-latency analysis

    Args:
        json_data : dict
        Dialogue data structure containing:
        - 'scan_id': str
            Unique identifier for the conversation session
        - 'system_prompt': str
            Initial system instruction for the dialogue
        - 'user_message': list[dict]
            Chronological dialogue turns, each with:
            * 'image_paths': list[str]
                Relative paths to visual context images
            * 'prompt': str
                User input text for this turn
        image_root : str
            Base directory for resolving relative image paths

        save_root : str
            Output directory where enhanced JSON will be saved
            (Files are named as "{scan_id}.json")

        model,tokenizer : object
           The InternVL model/tokenizer.

    Returns
        None
        Output is written to "{save_root}/{scan_id}.json" with:
        - Original input data
        - Added 'response' field for each turn
        - 'total_time' field recording processing duration
    """
    scan_id = data['scan_id']
    save_file_name = f'{save_root}/{scan_id}.json'
    if os.path.exists(save_file_name):
        return

    system_prompt = data['system_prompt']
    user_message_list = data['user_message']
    info_list = []
    for user_message in user_message_list:
        image_paths = user_message['image_paths']
        image_paths = [
            os.path.join(image_root, image_path) for image_path in image_paths
        ]
        for image_path in image_paths:
            assert os.path.exists(
                image_path), f'Image {image_path} does not exist'
        turn_id = user_message['turn_id']
        image_text = '\n'.join([
            f'Image-{i+1} for turn {turn_id}: <image>'
            for i in range(len(image_paths))
        ])
        question = f'{system_prompt}\n {image_text}\n {user_message["prompt"]}'
        info_list.append(image_paths + [question])

    start = time.time()
    results = internvl_multi_turn(model, info_list, tokenizer,
                                  generation_config)
    end = time.time()
    for i, turn_info in enumerate(data['user_message']):
        turn_info['response'] = results[i]
    data['total_time'] = end - start

    with open(save_file_name, 'w') as f:
        json.dump(data, f, indent=4)


def prepare_data(raw_data):
    """Converts a list of annotated data entries into organized multi-turn
    dialogues grouped by scan ID.

    This function processes individual interaction samples (each containing a
    scan_id) and aggregates them into complete multi-turn dialogue sessions,
    with one dialogue per scan_id. The output maintains a consistent structure
    for easy processing and analysis.

    Key processing steps:
    (1) Groups all entries with the same scan_id into a single dialogue session
    (2) Preserves the system_prompt as global context for each dialogue
    (3) Structures each turn with clear prompt-image pairing

    Args:
        raw_data (list[dict]): List of interaction samples where each contains:
        - scan_id (str): Unique scene/trajectory identifier
        - system_prompt (str): Initial system instruction for the dialogue
        - new_observations (list): Visual data associated with the turn
        - user_message (str): The user's input for this turn

    Returns:
        dict: Structured dialogues where:
        - Key: scan_id (str)
        - Value: dict containing:
        * 'system_prompt': str (shared across all turns)
        * 'scan_id': str (original identifier)
        * 'user_message': list[dict] (chronological turns), each with:
        - 'prompt': str (user input text)
        - 'image_paths': list (associated visual data paths)

    Note:
        Each scan_id corresponds to exactly one complete
        multi-turn dialogue session.
    """
    all_scan_data = {}
    for anno_ in raw_data:
        scan_id = anno_['scan_id']
        anno_['image_paths'] = anno_['new_observations']
        anno_['prompt'] = anno_['user_message']
        del anno_['new_observations'], anno_['user_message']
        if scan_id not in all_scan_data:
            all_scan_data[scan_id] = {
                'system_prompt': anno_['system_prompt'],
                'scan_id': anno_['scan_id'],
                'user_message': [],
            }
        all_scan_data[scan_id]['user_message'].append(anno_)
    return all_scan_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InternVL inference')

    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--anno_json_path',
                        type=str,
                        default='../data/OST_bench.json')
    parser.add_argument('--image_root', type=str, default='../data/images')
    parser.add_argument('--save_root',
                        type=str,
                        default='../outputs/InternVL-8B')
    parser.add_argument('--rank_num', type=int, default=2)
    parser.add_argument('--rank_index', type=int, default=0)
    args = parser.parse_args()

    model_path = args.model_path
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    all_datas = json.load(open(anno_json_path))
    all_scan_data = prepare_data(all_datas)
    all_scan_id_list = sorted(list(all_scan_data.keys()))
    num_scan = len(all_scan_id_list)
    num_per_rank = num_scan // args.rank_num
    rank_scan = [
        all_scan_id_list[i:i + num_per_rank]
        for i in range(0, num_scan, num_per_rank)
    ]
    rank_data = [
        all_scan_data[scan_id] for scan_id in rank_scan[args.rank_index]
    ]

    if '-8B' in model_path:
        device_map = split_model('InternVL2_5-8B')
    elif '-38B' in model_path:
        device_map = split_model('InternVL2_5-38B')
    else:
        device_map = split_model('InternVL2_5-78B')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=False)

    for data in tqdm(rank_data):
        process_data(data, model, tokenizer, image_root, save_root)
