import argparse
import json
import os
from functools import wraps

import mmengine
from tqdm import tqdm
from utils.openai_api import (get_content_groups_from_source_groups,
                              mimic_chat_budget)


def mmengine_track_func(func):

    @wraps(func)
    def wrapped_func(args):
        result = func(*args)
        return result

    return wrapped_func


OST_small_version = json.load(open('../meta-data/OST_small_version.json'))
OST_small_version = [scan.split('.')[0] for scan in OST_small_version]


def generate_multi_qa_gpt(input_dict):
    """Generates multiple Q&A pairs from source content using a specified GPT
    model.

    This function takes source content, processes it through
    a GPT model conversation simulation,
    and extracts the assistant's responses as raw annotations.

    Parameters:
    -----------
    input_dict : dict
        A dictionary containing the following keys:
        - 'system_prompt' : str
            The system prompt to initialize the conversation with the GPT model
        - 'user_message' : list or str
            The source content(s) to be processed, which will be converted into
            user messages
        - 'model_name' : str
            The name of the GPT model to use for generation

    Returns:
    --------
    list
        A list of raw annotations(strings) containing the assistant's responses
        from the conversation
    """
    system_prompt = input_dict['system_prompt']
    source_groups = input_dict['user_message']
    model_name = input_dict['model_name']
    content_groups = get_content_groups_from_source_groups(
        source_groups)  # high detail
    conversation = mimic_chat_budget(content_groups,
                                     system_prompt=system_prompt,
                                     model=model_name)
    raw_annotation = []
    for message in conversation:
        if message['role'] == 'assistant':
            raw_annotation.append(message['content'])

    return raw_annotation


@mmengine_track_func
def process_data(json_data, image_root, save_root, model_name='gpt-4o'):
    """This function takes conversation data for a specific scan ID, processes
    the image paths, generates responses for each dialogue turn using the
    specified GPT model, and saves the enhanced data (including responses) back
    to a JSON file.

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
        Root directory where the images are stored
            (will be prepended to image paths)
    save_root : str
        Directory where the output JSON file will be saved
    model_name : str, optional
        Name of the GPT model to use (default: 'gpt-4o')

    Returns:
    --------
    None
        The function saves the results to a file but doesn't
        return anything directly.
    """

    scan_id = json_data['scan_id']
    save_file_name = f'{save_root}/{scan_id}.json'
    if os.path.exists(save_file_name):
        return

    system_prompt = json_data['system_prompt']
    user_message_list = []
    for turn_info in json_data['user_message']:
        turn_info['image_paths'] = [
            os.path.join(image_root, image_path)
            for image_path in turn_info['image_paths']
        ]
        for image_path in turn_info['image_paths']:
            assert os.path.exists(image_path)
        print(turn_info['image_paths'])
        user_message_list.append(turn_info['image_paths'] +
                                 [turn_info['prompt']])

    input_dict = {
        'system_prompt': system_prompt,
        'user_message': user_message_list,
        'model_name': model_name,
    }

    try:
        results = generate_multi_qa_gpt(input_dict)

        for i, turn_info in enumerate(json_data['user_message']):
            turn_info['response'] = results[i]

        with open(os.path.join(save_file_name), 'w') as f:
            json.dump(json_data, f, indent=4)
    except:
        print(f'API Error processing {scan_id}')


def prepare_data(raw_data):
    """Converts a list of annotated data entries into organized multi-turn
    dialogues grouped by scan ID.

    This function processes individual interaction samples
    (each containing a scan_id) and aggregates them into complete
    multi-turn dialogue sessions, with one dialogue per scan_id. The
    output maintains a consistent structure for easyprocessing and
    analysis.

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
        Each scan_id corresponds to exactly one complete multi-turn
        dialogue session.
    """
    all_scan_data = {}
    for anno_ in raw_data:
        scan_id = anno_['scan_id']
        if scan_id not in OST_small_version:
            continue
        if scan_id not in all_scan_data:
            all_scan_data[scan_id] = {
                'system_prompt': anno_['system_prompt'],
                'scan_id': anno_['scan_id'],
                'user_message': [],
            }
        anno_['image_paths'] = anno_['new_observations']
        anno_['prompt'] = anno_['user_message']
        del anno_['new_observations'], anno_['user_message']
        all_scan_data[scan_id]['user_message'].append(anno_)

    return all_scan_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='proprietary baseline Inference')

    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--anno_json_path',
                        type=str,
                        default='../data/OST_bench.json')
    parser.add_argument('--image_root', type=str, default='../data/images')
    parser.add_argument('--save_root', type=str, default='../outputs/gpt-4o')
    parser.add_argument('--rank_num', type=int, default=1)
    args = parser.parse_args()

    model_name = args.model_name
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    all_datas = json.load(open(anno_json_path))

    all_scan_data = prepare_data(all_datas)

    all_tasks = [(all_scan_data[scan_id], image_root, save_root, model_name)
                 for scan_id in all_scan_data]
    if args.rank_num > 1:
        mmengine.utils.track_parallel_progress(process_data, all_tasks,
                                               args.rank_num)
    else:
        for task in tqdm(all_tasks):
            process_data(task)
