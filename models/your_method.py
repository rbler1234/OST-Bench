import os
import json
import random
from tqdm import tqdm
import argparse
import numpy as np

class Your_model:
    def __init__(self):
        pass
    
    def chat(self, prompt = None, imgs = None, history = None,system_prompt=None):
        if history is None:
            history = []
            
        #TODO: Implement your model approach!
        return response, history 



def process_data(json_data,image_root,save_root,your_model):
    """
    Processes multi-turn dialogue data by generating model responses and
    saving enhanced results to JSON.

    This function:
        (1) Processes image paths by joining with the provided root directory
        (2) Generates responses for each dialogue turn using the specified model
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

        your_model : object
            Model instance with:
            - chat(prompt: str, image_paths: list[str], 
            history: Any, system_prompt: str) -> tuple[str, Any]
            Method that generates responses given the inputs

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
    results = []
    start = time.time()
    for user_message in user_message_list:
        image_paths = user_message["image_paths"]
        image_paths = [os.path.join(image_root, image_path) for image_path in image_paths]
        for image_path in image_paths:
            assert os.path.exists(image_path), f"Image {image_path} does not exist"
        user_message['response'],history = your_model.chat(user_message["prompt"],
                                                    image_paths,history,system_prompt)
        
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
        if scan_id not in all_scan_data:
            all_scan_data[scan_id] = {'system_prompt': anno_['system_prompt'],
                                      'scan_id':anno_['scan_id'], 'user_message': []}
        anno_['image_paths'] = anno_['new_observations']
        anno_['prompt'] = anno_['user_message']
        del anno_['new_observations'],anno_['user_message']
        all_scan_data[scan_id]['user_message'].append(anno_)

    return all_scan_data
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Your method here')
    
    parser.add_argument('--anno_json_path', type=str, default='../data/OST_bench.json')
    parser.add_argument('--image_root', type=str, default='../data/images')
    parser.add_argument('--save_root', type=str, default='../outputs/your_method')
    args = parser.parse_args()
    
    model_name = args.model_name
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makedirs(save_root,exist_ok=True)
    all_datas = json.load(open(anno_json_path))
    all_scan_data = prepare_data(all_datas)
    
    your_model = Your_model()
    
    all_tasks = [(all_scan_data[scan_id],image_root,save_root,your_model) for scan_id in all_scan_data]
    
    for task in tqdm(all_tasks):
        process_data(task)
    
    
