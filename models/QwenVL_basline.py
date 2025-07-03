import argparse
import gc
import json
import os
import time

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen2VL:

    def __init__(
        self,
        model_path=None,
        max_new_tokens=8192,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
        )
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       min_pixels=min_pixels,
                                                       max_pixels=max_pixels)
        self.gen_config = {
            'max_new_tokens': max_new_tokens,
        }

    def parse_input(self, query=None, imgs=None, system_prompt=None):
        if imgs is None:
            messages = [{'role': 'user', 'content': query}]
            return messages

        if isinstance(imgs, str):
            imgs = [imgs]
        content = []
        if system_prompt is not None:
            content.append({'type': 'text', 'text': system_prompt})
        for img in imgs:
            content.append({'type': 'image', 'image': img})

        content.append({'type': 'text', 'text': query})
        messages = [{'role': 'user', 'content': content}]
        return messages

    def chat(self, query=None, imgs=None, history=None, system_prompt=None):
        if history is None:
            history = []

        user_query = self.parse_input(query, imgs, system_prompt)
        history.extend(user_query)

        text = self.processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = process_vision_info(history)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )

        inputs = inputs.to('cuda')
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        history.append({'role': 'assistant', 'content': response})

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        return response, history


def process_data(data, model, image_root, save_root):
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

        model : object
            Model instance with:
            - chat(prompt: str, image_paths: list[str],
            history: Any, system_prompt: str) -> tuple[str, Any]
           The QwenVL agent.

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
    history = None
    start = time.time()
    for user_message in user_message_list:
        image_paths = user_message['image_paths']
        image_paths = [
            os.path.join(image_root, image_path) for image_path in image_paths
        ]
        for image_path in image_paths:
            assert os.path.exists(
                image_path), f'Image {image_path} does not exist'
        user_message['response'], history = model.chat(user_message['prompt'],
                                                       image_paths, history,
                                                       system_prompt)

    end = time.time()
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
        Each scan_id corresponds to exactly one
        complete multi-turn dialogue session.
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
    parser = argparse.ArgumentParser(description='QwenVL inference')

    parser.add_argument(
        '--model_path',
        type=str,
        default='',
    )
    parser.add_argument('--anno_json_path',
                        type=str,
                        default='../data/OST_bench.json')
    parser.add_argument('--image_root', type=str, default='../data/images')
    parser.add_argument('--save_root', type=str, default='../outputs')
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

    qwen_agent = Qwen2VL(model_path)
    for data in tqdm(rank_data):
        process_data(data, qwen_agent, image_root, save_root)
