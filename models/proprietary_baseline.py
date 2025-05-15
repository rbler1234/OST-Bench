import os
import json
import random
from tqdm import tqdm
import mmengine
import argparse
import numpy as np
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups


OST_small_version = json.load(open('../meta-data/OST_small_version.json'))
OST_small_version = [scan.split('.')[0] for scan in OST_small_version]

def generate_multi_qa_gpt(input_dict):
    """Generates multiple Q&A pairs from source content using a specified GPT model.
    
    This function takes source content, processes it through a GPT model conversation simulation,
    and extracts the assistant's responses as raw annotations.
    
    Parameters:
    -----------
    input_dict : dict
        A dictionary containing the following keys:
        - 'system_prompt' : str
            The system prompt to initialize the conversation with the GPT model
        - 'user_message' : list or str
            The source content(s) to be processed, which will be converted into user messages
        - 'model_name' : str
            The name of the GPT model to use for generation
    
    Returns:
    --------
    list
        A list of raw annotations (strings) containing the assistant's responses from the conversation    
    """
    system_prompt = input_dict['system_prompt']
    source_groups = input_dict['user_message']
    model_name = input_dict['model_name']
    content_groups = get_content_groups_from_source_groups(source_groups)  # high detail
    conversation= mimic_chat_budget(content_groups, system_prompt=system_prompt,model_name=model_name)    
    raw_annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            raw_annotation.append(message["content"])
    
    return raw_annotation


def process_data(json_data,image_root,save_root,model_name='gpt-4o'):
    """
    This function takes conversation data for a specific scan ID, processes the image paths,
    generates responses for each dialogue turn using the specified GPT model, and saves
    the enhanced data (including responses) back to a JSON file.

    Parameters:
    -----------
    json_data : dict
        A dictionary containing the conversation data with the following structure:
        - 'scan_id': str - Unique identifier for the scan/session
        - 'system_prompt': str - System prompt for the GPT model
        - 'user_message': list[dict] - List of dialogue turns, each containing:
            * 'image_paths': list[str] - Relative paths to images for this turn
            * 'prompt': str - Text prompt for this turn
    image_root : str
        Root directory where the images are stored (will be prepended to image paths)
    save_root : str
        Directory where the output JSON file will be saved
    model_name : str, optional
        Name of the GPT model to use (default: 'gpt-4o')

    Returns:
    --------
    None
        The function saves the results to a file but doesn't return anything directly.    
    """
    
  
    scan_id = json_data['scan_id']
    save_file_name = f'{save_root}/{scan_id}.json'
    system_prompt = json_data['system_prompt']
    user_message_list = []
    for turn_info in json_data['user_message']:
        turn_info["image_paths"] = [os.path.join(image_root,image_path) for image_path in turn_info["image_paths"]]
        for image_path in turn_info["image_paths"]:
            assert os.path.exists(image_path)
        user_message_list.append(turn_info["image_paths"]+[turn_info["prompt"]])
        
    input_dict = {'system_prompt':system_prompt,'user_message':user_message_list,'model_name':model_name}
   
    try:
        
        results = generate_multi_qa_gpt(input_dict)
  
        for i,turn_info in enumerate(json_data['user_message']):
            turn_info['response'] = results[i]
  
        with open(os.path.join(save_file_name),'w') as f:
            json.dump(json_data,f,indent=4) 
        
    except:
        print(f"Error in processing {scan_id}")
            

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='proprietary baseline running')
    
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--anno_json_path', type=str, default='../data/OST_bench_v0.json')
    parser.add_argument('--image_root', type=str, default='../data/img/image_upload')
    parser.add_argument('--save_root', type=str, default='../results')
    args = parser.parse_args()
    
    model_name = args.model_name
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makdirs(save_root,exist_ok=True)
    all_datas = json.load(open(anno_json_path))
    for data in tqdm(all_datas):
        if data['scan_id'] not in OST_small_version:
            continue
        process_data(data,image_root,save_root,model_name)
