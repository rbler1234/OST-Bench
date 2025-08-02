import os
import json
import random
import mmengine
import numpy as np
import cv2
import argparse

system_prompt_dict = """Assume you are currently exploring a room where all objects are stationary. Over time, you change your position and orientation within the room and take images.
Now, I will engage you in a multi-turn dialogue (a total of {num_turns} turns). In each turn, I will provide you with {num_images} images taken from the beginning to the end of that turn, 
Please answer my questions based on your state(position/orientation) at the end (last image) of each turn."""
user_message_begin = """For the {turn_id} turn, these are the {num_images} images you took in chronological order. """

user_message_w_options= user_message_begin+"The question for this turn is: {Question} To answer this question, you need to combine image/coordinate information from past turns. Please choose the answer from {Options} based on history information."
user_message_w_num = user_message_begin+"The question for this turn is: {Question} To answer this question, you need to combine image/coordinate information from past turns. Please provide a numerical value as the result based on history information.The information I provided is sufficient for you to infer the value; do not refuse to answer!"
user_message_w_options_end= " Give me your answer and your reason in a JSON format({answer:str, reason:str}) "
user_message_w_num_end= " Give me your answer and your reason in a JSON format({answer:float/int, reason:str}) "

def name_mapping(raw_name):
    if raw_name == 'A_room-size(float)':
        return 'None'
    new_name = raw_name.replace('float','Estimation')

    if 'quantity' not in new_name:
        new_name = new_name.replace('int','Temporal-loc')
    else:
        new_name = new_name.replace('int','Counting')

    new_name = new_name.replace('option','Judgement')
    new_name = new_name.replace('A_object-','Agent_visible_info-')
    new_name = new_name.replace('AO_','Agent_object_spatial-')
    new_name = new_name.replace('A_','Agent_state-')
    return new_name  

def process_data(multi_qa,scan_id):
    input_dict = {}
    num_turns =len(multi_qa)
    num_images = len(multi_qa[0]["image_paths"])

    
    input_dict['scan_id'] = scan_id
    input_dict['system_prompt'] = system_prompt_dict.format(**{"num_images":num_images,"num_turns":num_turns})   
    input_dict['user_message'] = []

    for item_dict in multi_qa:
        
        # get info from the json
        turn_id = item_dict['turn_id']
        image_paths = []
        image_paths = item_dict["image_paths"]
          
        if item_dict['type'] != 'None':
            cand_ = random.choice(item_dict['candidate_'])
            Question = cand_['question']
            answer = cand_["answer"]
            if 'option' in cand_:
                Options = cand_["option"]
            else:
                Options = []
        else:
            Question = ''
            answer = 'I understand.'
            Options = []       
        # fill in the message template 
        
        if item_dict['type'] == 'None':
            text = user_message_begin.format(**{ "num_images": num_images,"turn_id":turn_id})+' There is no question for this turn.'
        else:
            if len(Options)>0:
                text = user_message_w_options.format(**{ "num_images": num_images,"turn_id":turn_id,
                                                            "Question":Question,"Options":Options})
                text += user_message_w_options_end
            else:    
                text = user_message_w_num.format(**{ "num_images": num_images,"turn_id":turn_id,"Question":Question})
                text += user_message_w_num_end
        
        # fill in the image input    
        user_message_list =  image_paths

        input_dict['user_message'].append({'turn_id':item_dict['turn_id'],'type':name_mapping(item_dict['type']),'origin_question':Question,'option':Options,'answer':answer,
                                            'image_paths':user_message_list,'prompt':text})
    return input_dict

if __name__ == '__main__':
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Step2')
    
    parser.add_argument('--split', type=str,default='scannet')
    args = parser.parse_args()
    
    scan_split = args.split
    source_dir_list = [f'./data/step_1/{scan_split}']

    target_dir = f'./data/step_2'
    os.makedirs(target_dir,exist_ok=True)
    for source_dir in source_dir_list:
   
        for json_file in tqdm(os.listdir(source_dir)):
            json_file_path = os.path.join(source_dir,json_file)
            scan_id = json_file.split('.')[0]

          
            multi_qa = json.load(open(json_file_path,'r'))
            results = process_data(multi_qa,scan_id)
            with open(f'{target_dir}/{scan_id}.json','w') as f:
                json.dump(results,f,indent=4)