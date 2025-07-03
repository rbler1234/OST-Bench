import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
import torch
from PIL import Image
import math
import time
import json
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, 
                    has_image: bool = False, max_len=10048, 
                system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens \
                + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) \
            + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

        
    
def process_data(data,model,tokenizer, image_processor, context_len, image_root,save_root):
    """
    Processes multi-turn dialogue data by generating model responses and saving 
    enhanced results to JSON.

    This function:
        (1) Processes image paths by joining with the provided root directory
        (2) Generates responses for each dialogue turn using the specified model
        (3) Maintains conversation history across turns, LLaVA models lack multi-turn support, 
            so dialogue history is handled via interleaving
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

        model,tokenizer, image_processor, context_len

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
    all_images = []
    for user_message in user_message_list:
        image_paths = user_message["image_paths"]
        image_paths = [os.path.join(image_root, image_path) for image_path in image_paths]
        for image_path in image_paths:
            assert os.path.exists(image_path), f"Image {image_path} does not exist"
        all_images.extend(image_paths)
        image_num = len(image_paths)

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], system_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    start = time.time()
    all_response = []
    for test_idx in range(len(user_message_list)):
        
        input_text = f"""Assume you are currently exploring a room where all objects are stationary. Over time, you change your position and orientation within the room and take images while exploring.
        Now, I will engage you in a multi-turn dialogue(totally {test_idx+1} turns). In each turn, I will provide you with {image_num} images taken from the beginning to the end of each turn. 
        And I will ask a question at the end of the final turn ({test_idx+1} turn), please answer my question based on your state(position/orientation) at the end (last image) of the final turn."""
        
        for index in range(test_idx):
        
            input_text += f'For the {index+1} turn, these are the {image_num} images you took in chronological order: '+'<image> '*image_num+ '\n'
            
        input_text+=user_message_list[test_idx]['prompt'].split('.')[0]+ '<image> '*image_num+user_message_list[test_idx]['prompt'].split('chronological order')[1]
        input_text = input_text.split('Give me your answer and')[0]+'Only give me your answer!'
        
        input_ids = preprocess_qwen([{'from':'human','value':input_text},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

        image_tensors = []
        for image_file in all_images:
            image = Image.open(image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=8192,
                use_cache=True)

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        all_response.append(outputs)
    
    for i,turn_info in enumerate(data['user_message']):
        turn_info['response'] = all_response[i]

    end = time.time()    
    data['total_time'] = end-start 
    
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
        Each scan_id corresponds to exactly one complete multi-turn dialogue session.
    """
    all_scan_data = {}
    for anno_ in raw_data:
        scan_id = anno_['scan_id']
        anno_['image_paths'] = anno_['new_observations']
        anno_['prompt'] = anno_['user_message']
        del anno_['new_observations'],anno_['user_message']
        if scan_id not in all_scan_data:
            all_scan_data[scan_id] = {'system_prompt': anno_['system_prompt'],'scan_id':anno_['scan_id'], 'user_message': []}
        all_scan_data[scan_id]['user_message'].append(anno_)
    return all_scan_data
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='LLaVA inference')
    
    parser.add_argument('--model_path', type=str,default='/mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LLaVA-Next/pretrain/llava-onevision-qwen2-7b-ov-chat')
    parser.add_argument('--anno_json_path', type=str, default='../data/OST_bench.json')
    parser.add_argument('--image_root', type=str, default='../data/images')
    parser.add_argument('--save_root', type=str, default='../outputs/llava_onevision_7b_new_env')
    parser.add_argument('--rank_num', type=int, default=2)
    parser.add_argument('--rank_index', type=int, default=0)
    args = parser.parse_args()
    
    model_path = args.model_path
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makedirs(save_root,exist_ok=True)
    
    
    all_datas = json.load(open(anno_json_path))
    all_scan_data = prepare_data(all_datas)
    all_scan_id_list = sorted(list(all_scan_data.keys()))
    num_scan = len(all_scan_id_list)
    num_per_rank = num_scan // args.rank_num
    rank_scan = [all_scan_id_list[i:i+num_per_rank] for i in range(0, num_scan, num_per_rank)]
    rank_data = [all_scan_data[scan_id] for scan_id in rank_scan[args.rank_index]]
   
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    
    for data in tqdm(rank_data):
        process_data(data,model,tokenizer, image_processor, context_len, image_root,save_root)  
