import argparse
import os
import json
import numpy as np
import random
import torch
import itertools
from itertools import permutations
from copy import deepcopy
from utils.template_utils import direction_judging, item_rounding,distance_between_pcd
from utils.all_info import all_translate,common_type
from tqdm import tqdm
import mmengine
import math



# sample gen setting
DIVERSITY_EACH_SCAN = 1
ANGLE_THRESHOLD = 30
DISTANCE_THRESOLD = 0.5
DISTANCE_COMPARE_RATIO = 0.4
SiZE_COMPARE_RATIO = 0.2
TURN_BIAS = 1

# 不太好计数的物体
not_countable = ['curtain','steps','step','cabinet','socket','book','rail','food','grass','label','ledge','machine','magazine','paper','stairs','wood','box']
# 一些没有额外描述不太好确定的物体
not_refer = ['box','cabinet','dress','pack','wrap','socket','paper','roof','case','pipe','picture','cup','bottle','plug','eraser','pen','hamper','ledge','bag','basin','toy','furniture']
# 一些直接排除掉的物体
EXCULED_OBJECT_TYPE = ['object','ceiling','wall','floor','column','doorframe','socket','plug','label']




OUTTRUN_TYPE =['AO_direction(option1)','AO_direction(option2)','AO_direction(option3)','AO_distance(option1)','AO_distance(option2)','AO_distance(option3)','AO_distance(int)','AO_direction(int)','AO_distance(float)','AO_direction(float)',
'A_object-order(option)','A_object-existence(option)','A_object-existence(int1)','A_object-existence(int2)','A_object-quantity(int)','A_object-diversity(option)','A_room-size(float)',
'A_position(option)','A_position(float)','A_orientation(option)','A_orientation(float)']


# distribution setting


WEIGHT_DICT = {k:1.0 for k in OUTTRUN_TYPE}
WEIGHT_DICT={'AO_direction(option1)': 2.0, 
             'AO_direction(option2)': 3.0,
             'AO_direction(option3)':3.0,
             'AO_distance(option1)': 1.5, 
             'AO_distance(option2)': 2.0,
             'AO_distance(option3)': 1.0,
             'AO_distance(float)': 2.0, 
             'AO_direction(float)': 1.5, 
             'AO_distance(int)': 1.0, 
             'AO_direction(int)': 1.0, 

             
             'A_object-order(option)': 3.2, 
             'A_object-existence(int1)': 1.5,
             'A_object-existence(int2)': 2.0,
             'A_object-existence(option)': 0.3, 
             'A_object-quantity(int)': 0.5, 
             'A_object-diversity(option)': 6.0,
             'A_room-size(float)':0.5,
             

             'A_position(option)':1.0,'A_position(float)':1.0,'A_orientation(option)':1.0,'A_orientation(float)':1.0}

no_in_first = ['A_room-size(float)']
for key in WEIGHT_DICT:
    if 'A_object' in key:
        WEIGHT_DICT[key]*=0.5
    
    



class QA_Sizer():
    def __init__(self, weight_dict) -> None:
        self.weight_dict = weight_dict
        self.history = []
        self.type_cnt = {k:0 for k in self.weight_dict}
        
    def choose_from_samples(self, type_samples_dict):
        # filter in the history
        # it's ok if the question is the same but the anser is not the same
        for type_ in type_samples_dict:
            type_samples_dict[type_] = [k for k in type_samples_dict[type_] if (type_,(k[0],k[1])) not in self.history]
        type_pools = []
        for type_ in type_samples_dict:
            if len(type_samples_dict[type_])>=2 and (type_ not in no_in_first or len(self.history)>=1):
                type_pools.append(type_)
                
            self.type_cnt[type_]+=len(type_samples_dict[type_])
        
        type_weight = [self.weight_dict[type_] for type_ in type_pools]
        cnt_weight = sum(type_weight)
        if cnt_weight==0:
            return 'None',('There are no questions for you to answer in this turn.',''),[]
        type_weight = [k/cnt_weight for k in type_weight]  
              
        type_choice =  random.choices(type_pools, weights=type_weight, k=1)[0]

        sample_choice = random.choice(type_samples_dict[type_choice])
        self.history.append((type_choice,(sample_choice[0],sample_choice[1])))
        return type_choice,sample_choice, type_samples_dict[type_choice]

class Info_of_turns():
    def __init__(self, scan_id = 'scene0000_00', \
        sample_id = 0, start_turn = -1, end_turn = -1) -> None:
        if start_turn==-1 or end_turn==-1:
            num_turns = len([file for file in os.listdir(os.path.join(DATA_DIR, f"{scan_id}_{sample_id}")) if file[-4:]=='json'])

            start_turn = 0
            end_turn = min(num_turns-1,9)
    
        self.total_turns = end_turn - start_turn + 1
        self.turn_info = {}
        self.total_visible = {}
        self.prefix_visible = {}
        self.img_fix = {}
        pcd_info = torch.load(f'{PCD_DIR}/{scan_id}.pth')
        self.obj_pcds = {}
        pcds = pcd_info[0]
        instance_labels = pcd_info[-1]
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i
            if len(pcds[mask]) > 0:
                self.obj_pcds.update({i: pcds[mask]})
        
        for turn_id in range(start_turn, end_turn+1):
            info_json_path = os.path.join(DATA_DIR, f"{scan_id}_{sample_id}",f"info_{turn_id}.json")
            self.turn_info[turn_id] = self.load_info(info_json_path,turn_id)
            self.total_visible.update(self.turn_info[turn_id]["visible_object_info"])
            self.prefix_visible[turn_id] = deepcopy(self.total_visible)
        
    def filter_extype_set(self,object_set):
        if isinstance(object_set,dict):
            new_dict = {k:v for k,v in object_set.items() if k.split('_')[0] not in EXCULED_OBJECT_TYPE}   
            return  new_dict
        if isinstance(object_set,list):
            new_list = [item for item in object_set if item.split('_')[0] not in EXCULED_OBJECT_TYPE]
            return  new_list
    def load_info(self, info_file, turn_id):
        info_dict = {}
        with open(info_file, 'r') as f:
            info_data = json.load(f) 
        info_dict['turn_id'] = turn_id   
        info_dict['images'] = [item["image_path"] for item in info_data["trajory_info"]]
        info_dict['position'] = [item["pos"] for item in info_data["trajory_info"]] 
        info_dict['position_wz'] = [item["pos_xyz"] for item in info_data["trajory_info"]] 
        info_dict['orientation'] = [item["front"] for item in info_data["trajory_info"]]
        info_dict['visible_object_info'] = self.filter_extype_set(info_data["map_info"])
  
        info_dict['countable_info'] = self.filter_extype_set(info_data["countable_info"])
        
        
        # fix bug for arkitscenes
        if "turn image" in info_data["trajory_info"][0]:
            info_dict['image_rotate'] = info_data["trajory_info"][0]["turn image"]
        else:
            info_dict['image_rotate'] = 180

    
        return info_dict

# TYPE-1: Agent-object spatial

def generate_for_AO_direction_option_2d(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-direction1: 基于当前确定不可见物体方位，选择，二选一题
    '''
    sample_pools = []
    for object_name in disappear_objects:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names:
            continue
        if object_name not in countable_list:
            continue
        object_id = int(object_id)
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        skip_flag = False
        
          
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue
        v1 = np.array(agent_front[current_turn_id])
        v2 = np.array(current_total_visible[object_name]['pos'])- np.array(agent_pos[current_turn_id])
        
        question = f'Is {object_caption} to your left/right now?'
        answer = direction_judging(v1,v2,mode='L&R')
        option = ['left','right']
        info = {object_name:object_caption}
        if answer!='':
            sample_pools.append((question,answer,option,info))
    return sample_pools
def generate_for_AO_direction_option_4d(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    ''' 4 direction
    '''
    sample_pools = []
    for object_name in disappear_objects:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names:
            continue
        if object_name not in countable_list:
            continue
        object_id = int(object_id)
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        skip_flag = False
        
          
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue
        v1 = np.array(agent_front[current_turn_id])
        v2 = np.array(current_total_visible[object_name]['pos'])- np.array(agent_pos[current_turn_id])
        
        question = f'Which direction is {object_caption} in relative to you now: front left, front right, rear left, or rear right?'
        answer = direction_judging(v1,v2,mode='L&R&F&R')
        option = ['front-right','rear-right','rear-left','front-left']
        info = {object_name:object_caption}
        if answer!='':
            sample_pools.append((question,answer,option,info))
    return sample_pools
def generate_for_AO_direction_option_3d(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    ''' 3 direction
    '''
    sample_pools = []
    for object_name in disappear_objects:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names:
            continue
        if object_name not in countable_list:
            continue
        object_id = int(object_id)
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        
        skip_flag = False
        
          
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue
        
        v1 = np.array(agent_front[current_turn_id])
        v2 = np.array(current_total_visible[object_name]['pos'])- np.array(agent_pos[current_turn_id])
        
        question = f'Which direction is {object_caption} in relative to you now— left, right or behind?'
        answer = direction_judging(v1,v2,mode='L&R&B')
        option = ['left', 'right','back']
        if answer!='':
            sample_pools.append((question,answer,option))
    return sample_pools
def generate_for_AO_direction_option_3o(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-direction2：基于当前确定两个物体方位关系，选择，三选一题
    '''
    sample_pools = []
    tmp_direction_dict = {}
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        object_id = int(object_id)
        
         
        v1 = np.array(agent_front[current_turn_id])
        v2 = np.array(current_total_visible[object_name]['pos'])- np.array(agent_pos[current_turn_id])
        answer = direction_judging(v1,v2,mode='L&R')
        if answer!='':
            if tmp_direction_dict.get(answer,None) is None:
                tmp_direction_dict[answer] = []
            tmp_direction_dict[answer].append(object_name)
    all_possible_names = []
    #print(tmp_direction_dict)
    for answer in tmp_direction_dict:
        all_possible_names.extend(tmp_direction_dict[answer])
    raw_tuple_set = list(itertools.combinations(all_possible_names, 3))
    
    for tuple_ in raw_tuple_set:
        cnt_answer = {k:0 for k in tmp_direction_dict.keys()}
        answer_side = ''
        max_cnt = 0
        for item_ in tuple_:
            for answer_ in tmp_direction_dict:
                if item_ in tmp_direction_dict[answer_]:
                    cnt_answer[answer_]+=1
        #print(cnt_answer)
        for answer_ in cnt_answer:
            if cnt_answer[answer_]>max_cnt:
                answer_side = answer_
                max_cnt = cnt_answer[answer_]            
        if any(cnt_answer[k]==3 for k in tmp_direction_dict):
      
            continue
        
        skip_flag = False
        for object_name in tuple_:
            object_type = object_name.split('_')[0]
            if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                skip_flag = True
        if skip_flag:
            continue
        if not any(object_name in disappear_objects for object_name in tuple_) or all(object_name in disappear_objects for object_name in tuple_):
        
            continue
        
        text_question = ''
        info = {}
        for object_name in tuple_:
            object_type,object_id = object_name.split("_")
            object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
            info[object_name] = object_caption
            text_question+= '\"'+object_caption+'\"'+', '
        text_question = text_question[:-2]+'. '
        text_question = f'Which two objects are on the same side of you now? {text_question}'
        text_answer = ''
        text_option = []
        for object_name in tuple_:
            object_type,object_id = object_name.split("_")
            object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
            if object_name in tmp_direction_dict[answer_side]:
                
                text_answer+= object_caption+', '
            text_option.append(object_caption) 
        text_answer = text_answer[:-2]
        option_A = text_option[0]+', '+ text_option[1]
        option_B = text_option[0]+', '+ text_option[2]
        option_C = text_option[1]+', '+ text_option[2]
        options = [option_A,option_B,option_C]
        
        sample_pools.append((text_question,text_answer,options,info))
    return sample_pools

def generate_for_AO_distance_option_2o2a(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-distance1: 两个物体与两时刻agent间距离，选择，四选一题
       here's a bug: z axis isn't be taken care of
    '''
    
    sample_pools = []
    unqine_object_names = []
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    raw_tuple_set = list(itertools.combinations(unqine_object_names , 2))
    for past_turn_id in range(current_turn_id):
        for object_name1, object_name2 in raw_tuple_set: 
            if object_name1==object_name2:
                continue 
            object_type1,object_id1 = object_name1.split("_")
            object_id1 = int(object_id1)
            object_caption1 = current_total_visible[object_name1].get("caption",f"the {object_type1}").replace("a ","the ")
            object_type2,object_id2 = object_name2.split("_")
            object_id2 = int(object_id2)
            object_caption2 = current_total_visible[object_name2].get("caption",f"the {object_type2}").replace("a ","the ")
            skip_flag = False
            for object_name in [object_name1,object_name2]:
                object_type = object_name.split('_')[0]
                if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                    skip_flag = True
            if skip_flag:
                continue
            
            if object_id1 not in obj_pcds or object_id2 not in obj_pcds:
                continue
            current_agent_pos = np.array(agent_pos_wz[current_turn_id])
            past_agent_pos = np.array(agent_pos_wz[past_turn_id])
            distance1_1 = distance_between_pcd(obj_pcds[object_id1],past_agent_pos)
            distance1_2 = distance_between_pcd(obj_pcds[object_id1],current_agent_pos)
            distance2_1 = distance_between_pcd(obj_pcds[object_id2],past_agent_pos)
            distance2_2 = distance_between_pcd(obj_pcds[object_id2],current_agent_pos)
            if (distance1_1/distance1_2<(1+DISTANCE_COMPARE_RATIO) and distance1_2/distance1_1<(1+DISTANCE_COMPARE_RATIO)) or( distance2_1/distance2_2<(1+DISTANCE_COMPARE_RATIO) and distance2_2/distance2_1<(1+DISTANCE_COMPARE_RATIO)):
                continue

            question = f'By the end of the {past_turn_id+TURN_BIAS} turn, were you getting closer to or farther away from these two objects: X: \"{object_caption1}\", Y: \"{object_caption2}\" ?'
            option = ['closer to both X and Y','farther away from both X and Y','closer to X and farther away from Y','closer to Y and farther away from X']
            if distance1_1<distance1_2 and distance2_1<distance2_2:
                answer = 'farther away from both X and Y'
            elif distance1_1>distance1_2 and distance2_1>distance2_2:
                answer = 'closer to both X and Y'
            elif distance1_1>distance1_2 and distance2_1<distance2_2:
                answer = 'closer to X and farther away from Y'
            elif distance1_1<distance1_2 and distance2_1>distance2_2:
                answer = 'closer to Y and farther away from X'
            info = {object_name1:object_caption1,object_name2:object_caption2}
            sample_pools.extend([(question,answer,option,info)])
    return sample_pools
def generate_for_AO_distance_option_3o(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-distance1: 基于当前比较与两个物体间距离，选择，二选一题，至少一个不可见
       here's a bug: z axis isn't be taken care of
    '''
    def random_sample_with_inclusion(lst, element, sample_size=3):
        sample = [element]
        remaining_sample_size = sample_size - 1
        remaining_elements = [x for x in lst if x != element]
        additional_samples = random.sample(remaining_elements, remaining_sample_size)
        sample.extend(additional_samples)
        random.shuffle(sample)
        return sample
    
    sample_pools = []
    
    unqine_object_names = []
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    raw_tuple_set = list(itertools.combinations(unqine_object_names , 3))
    for tuple_ in raw_tuple_set:    
        object_name1 = tuple_[0]
        object_name2 = tuple_[1]
        object_name3 = tuple_[2]
        object_type1 = object_name1.split('_')[0]
        object_id1 = int(object_name1.split('_')[1])
        object_type2 = object_name2.split('_')[0]
        object_id2 = int(object_name2.split('_')[1])
        object_type3 = object_name3.split('_')[0]
        object_id3 = int(object_name3.split('_')[1])
        skip_flag = False
        for object_name in [object_name1,object_name2,object_name3]:
            object_type = object_name.split('_')[0]
            if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                skip_flag = True
        if skip_flag:
            continue
        object_caption1 = current_total_visible[object_name1].get("caption",f"the {object_type1}").replace("a ","the ")
        object_caption2 = current_total_visible[object_name2].get("caption",f"the {object_type2}").replace("a ","the ")
        object_caption3 = current_total_visible[object_name3].get("caption",f"the {object_type3}").replace("a ","the ")
    

        if object_id1 not in obj_pcds or object_id2 not in obj_pcds or object_id3 not in obj_pcds:
            continue
        current_agent_pos = np.array(agent_pos_wz[current_turn_id])
        distance_1 = distance_between_pcd(obj_pcds[object_id1],current_agent_pos)
        distance_2 = distance_between_pcd(obj_pcds[object_id2],current_agent_pos)
        distance_3 = distance_between_pcd(obj_pcds[object_id3],current_agent_pos)
      
        
        cnt = 0
        for obj_name in [object_name1,object_name2,object_name3]:
            if obj_name in disappear_objects:
                cnt +=1
        if cnt<3:
            continue

        max_index = -1
        min_index = -1
        distance_list = [distance_1,distance_2,distance_3]
        for index_1 in range(3):
            max_flag = True
            min_flag = True
            for index_2 in range(3):
                if index_1==index_2:
                    continue
                if (1+DISTANCE_COMPARE_RATIO)*distance_list[index_1]>distance_list[index_2]:
                    min_flag = False
                if distance_list[index_1]<(1+DISTANCE_COMPARE_RATIO)*distance_list[index_2]:
                    max_flag = False
            if max_flag:
                max_index = index_1
            if min_flag:
                min_index = index_1
        if max_index==-1 and min_index==-1:
            continue
        object_caption_list = [object_caption1,object_caption2,object_caption3]
        if max_index>-1:
            question = f'Among these three objects(\"{object_caption1}\",\"{object_caption2}\",\"{object_caption3}\" ), which one is the farthest from you?'
            options = [object_caption1,object_caption2,object_caption3]
            answer = f'{object_caption_list[max_index]}'
            info = {object_name1:object_caption1,object_name2:object_caption2,object_name3:object_caption3}
            sample_pools.append((question,answer,options,info))
        if min_index>-1:
            question = f'Among these three objects(\"{object_caption1}\",\"{object_caption2}\",\"{object_caption3}\" ), which one is the closet to you?'
            options = [object_caption1,object_caption2,object_caption3]
            answer = f'{object_caption_list[min_index]}'
            info = {object_name1:object_caption1,object_name2:object_caption2,object_name3:object_caption3}
            sample_pools.append((question,answer,options,info))
        
        
    
    return sample_pools
def generate_for_AO_distance_option_2o(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-distance1: 基于当前比较与两个物体间距离，选择，二选一题，至少一个不可见
       here's a bug: z axis isn't be taken care of
    '''
    
    sample_pools = []
    
    unqine_object_names = []
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    raw_tuple_set = list(itertools.combinations(unqine_object_names , 2))
    for tuple_ in raw_tuple_set:    
        object_name1 = tuple_[0]
        object_name2 = tuple_[1]
        object_type1 = object_name1.split('_')[0]
        object_id1 = int(object_name1.split('_')[1])
        object_type2 = object_name2.split('_')[0]
        object_id2 = int(object_name2.split('_')[1])
        object_caption1 = current_total_visible[object_name1].get("caption",f"the {object_type1}").replace("a ","the ")
        object_caption2 = current_total_visible[object_name2].get("caption",f"the {object_type2}").replace("a ","the ")
        # object_pos1 = np.array(current_total_visible[object_name1]['pos'])
        # object_pos2 = np.array(current_total_visible[object_name2]['pos'])
        if object_id1 not in obj_pcds or object_id2 not in obj_pcds:
            continue
        current_agent_pos = np.array(agent_pos_wz[current_turn_id])
        distance_1 = distance_between_pcd(obj_pcds[object_id1],current_agent_pos)
        distance_2 = distance_between_pcd(obj_pcds[object_id2],current_agent_pos)
        skip_flag = False
        for object_name in [object_name1,object_name2]:
            object_type = object_name.split('_')[0]
            if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                skip_flag = True
        if skip_flag:
            continue    
        if object_name1 not in disappear_objects and object_name2 not in disappear_objects:
            continue
        if distance_1/distance_2<(1+DISTANCE_COMPARE_RATIO) and distance_2/distance_1<(1+DISTANCE_COMPARE_RATIO):
            continue
        option = [object_caption1,object_caption2]
        if distance_1<distance_2:
            question1 = f'Which of these two objects is farther away from you now? \"{object_caption1}\" or \"{object_caption2}\"'
            answer1 = object_caption2
            question2 = f'Which of these two objects is closer to you now? \"{object_caption1}\" or \"{object_caption2}\"'
            answer2 = object_caption1
            sample_pools.extend([(question1,answer1,option),(question2,answer2,option)])
        else:
            question1 = f'Which of these two objects is farther away from you now? \"{object_caption1}\" or \"{object_caption2}\"'
            answer1 = object_caption1
            question2 = f'Which of these two objects is closer to you now? \"{object_caption1}\" or \"{object_caption2}\"'
            answer2 = object_caption2
            sample_pools.extend([(question1,answer1,option),(question2,answer2,option)])
    
    return sample_pools
def generate_for_AO_distance_option_2a(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-distance1: 物体与两时刻agent间距离，选择，二选一题
       here's a bug: z axis isn't be taken care of
    '''
    
    sample_pools = []
    unqine_object_names = []
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    
    for past_turn_id in range(current_turn_id):
        for object_name in unqine_object_names:  
            object_type,object_id = object_name.split("_")
            object_id = int(object_id)
            object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
            object_pos = np.array(current_total_visible[object_name]['pos'])
            if object_id not in obj_pcds:
                continue
            current_agent_pos = np.array(agent_pos_wz[current_turn_id])
            past_agent_pos = np.array(agent_pos_wz[past_turn_id])
            distance_1 = distance_between_pcd(obj_pcds[object_id],current_agent_pos)
            distance_2 = distance_between_pcd(obj_pcds[object_id],past_agent_pos)
            if distance_1/distance_2<(1+DISTANCE_COMPARE_RATIO) and distance_2/distance_1<(1+DISTANCE_COMPARE_RATIO):
                continue
            skip_flag = False
            
            if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                skip_flag = True
            if skip_flag:
                continue
            option = ['Closer','Farther']
            if distance_1<distance_2:
                question1 = f'Compared to your position at the end of the {past_turn_id+TURN_BIAS} turn, are you now closer or farther away from \"{object_caption}\"?'
                answer1 = 'Closer'
                info = {object_name:object_caption}
                sample_pools.extend([(question1,answer1,option,info)])
                #print(np.linalg.norm(object_pos-current_agent_pos),np.linalg.norm(object_pos-past_agent_pos))
            else:
                question1 = f'Compared to your position at the end of the {past_turn_id+TURN_BIAS} turn, are you now closer or farther away from \"{object_caption}\"?'
                answer1 = 'Farther'
                info = {object_name:object_caption}
                sample_pools.extend([(question1,answer1,option,info)])
                
            # if answer1=='Farther':
            #     print(question1,answer1)
    
    return sample_pools


def generate_for_AO_distance_float(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for easy-SG-distance1: 物体与agent间的水平距离，直接问，目前不可见
    '''
    
    sample_pools = []
    unqine_object_names = []
    for object_name in disappear_objects:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    
    for object_name in unqine_object_names:  
        object_type,object_id = object_name.split("_")
        object_id =int(object_id)
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        object_pos = np.array(current_total_visible[object_name]['pos'])
        current_agent_pos = np.array(agent_pos[current_turn_id])
        if object_id not in obj_pcds:
            continue
        skip_flag = False
            
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue    
        distance_1 = distance_between_pcd(obj_pcds[object_id][:,:2],current_agent_pos)
        if distance_1<DISTANCE_THRESOLD:
            continue
        question = f'Please recall \"{object_caption}\", what is the horizontal distance between you and this object now (in meter)?'
        answer = item_rounding(distance_1)
        info = {object_name:object_caption}
        sample_pools.extend([(question,answer,info)])

    return sample_pools
def generate_for_AO_direction_float(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for hard_SG_direction: 物体的方位，直接问，目前不可见
    '''
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product == 0:
            cross_product = -1
        raw_angle = np.arccos(cos_theta)*180/np.pi * -np.sign(cross_product)
        if raw_angle < 0:
            raw_angle+=360.0
        return raw_angle
    sample_pools = []
    unqine_object_names = []
    for object_name in disappear_objects:
        object_type,object_id = object_name.split("_")
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_names.append(object_name) 
    
    for object_name in unqine_object_names:  
        object_type,object_id = object_name.split("_")
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        object_pos = current_total_visible[object_name]['pos']
        current_agent_pos = np.array(agent_pos[current_turn_id])
        current_agent_front = np.array(agent_front[current_turn_id])
        vector = np.array(object_pos-current_agent_pos)
        angle =  get_angle(current_agent_front,vector) 
        skip_flag = False
            
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue  
        if angle>180:
            angle = angle-360
        if abs(angle)<ANGLE_THRESHOLD:
            continue
        question = f'Based on your current orientation, in what {"clockwise" if angle>0 else "counterclockwise"} direction (in degrees) is \"{object_caption}\" from your position?'
        answer = item_rounding(abs(angle))
        info = {object_name:object_caption}
        sample_pools.extend([(question,answer,info)])

    return sample_pools

def generate_for_AO_distance_int(obj_pcds,current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for no_TG_state1: 位置的距离某物体最近/最远
    '''
    samples = []
    if current_turn_id==0:
        return []
    for object_name in current_total_visible:
        object_type = object_name.split('_')[0]
        object_id = int(object_name.split('_')[1])
        if object_type not in unqine_names:
            continue
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        object_pos = current_total_visible[object_name]['pos']
        
        if object_id not in obj_pcds:
            continue
        
        agent_distance = {turn_id: 
            np.min([distance_between_pcd(obj_pcds[object_id][:,:2],np.array(agent_pos))  for agent_pos in current_turn_info[turn_id]["position"]])
            for turn_id in current_turn_info.keys()}
        order_dict = {k:v for (k,v) in agent_distance.items()}
        sorted_list = sorted(order_dict.items(), key=lambda item: item[1])
        order_dict = {k:v for (k,v) in sorted_list}
        
        distance_order = list(order_dict.keys())
        if order_dict[distance_order[0]]*(1+DISTANCE_COMPARE_RATIO)<order_dict[distance_order[1]]:
            question = f'In which turn were you closest to \"{object_caption}\"? '
            answer = distance_order[0]+TURN_BIAS 
            info = {object_name:object_caption}
            samples.append((question,answer,info))
    for object_name in current_total_visible:
        object_type = object_name.split('_')[0]
        object_id = int(object_name.split('_')[1])
        if object_type not in unqine_names:
            continue
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        object_pos = current_total_visible[object_name]['pos']
        
        if object_id not in obj_pcds:
            continue
        skip_flag = False
            
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue              
    
        
        agent_distance = {turn_id: 
            np.max([distance_between_pcd(obj_pcds[object_id][:,:2],np.array(agent_pos)) for agent_pos in current_turn_info[turn_id]["position"]])
            for turn_id in current_turn_info.keys()}
        order_dict = {k:v for (k,v) in agent_distance.items()}
        sorted_list = sorted(order_dict.items(), key=lambda item: item[1])
        order_dict = {k:v for (k,v) in sorted_list}
        
        distance_order = list(order_dict.keys())
        if order_dict[distance_order[-2]]*(1+DISTANCE_COMPARE_RATIO)<order_dict[distance_order[-1]]:
            question = f'In which turn were you farthest from \"{object_caption}\"? '
            answer = distance_order[-1]+TURN_BIAS 
            info = {object_name:object_caption}
            samples.append((question,answer,info))
    return samples
def generate_for_AO_direction_int(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list):
    '''template for no_TG_state2: 位置的特殊性事件：方位同侧
    '''
    sample_pools = []
    if current_turn_id==0:
        return []
    unqine_object_list_total =[]
    for object_name in current_total_visible:
        object_type = object_name.split('_')[0]
        
        if object_type not in unqine_names or object_name not in countable_list:
            continue
        unqine_object_list_total.append(object_name)
    raw_tuple_set1 = list(itertools.combinations(unqine_object_list_total, 2))
    raw_tuple_set2 = list(itertools.combinations(unqine_object_list_total, 3))
    
    for object_name1, object_name2 in raw_tuple_set1:    
        object_type1 = object_name1.split('_')[0]
        object_caption1 = current_total_visible[object_name1].get("caption",f"the {object_type1}").replace("a ","the ")
        object_pos1 = current_total_visible[object_name1]['pos']
        
        object_type2 = object_name2.split('_')[0]
        object_caption2 = current_total_visible[object_name2].get("caption",f"the {object_type2}").replace("a ","the ")
        object_pos2 = current_total_visible[object_name2]['pos']
        skip_flag = False
        for object_name in [object_name1,object_name2]:
            object_type = object_name.split('_')[0]
            if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                skip_flag = True
        if skip_flag:
            continue
        turn_direction_info = {'left':{},'right':{},'':{}}
        for turn_id in range(current_turn_id+1):
            turn_direction_info['left'][turn_id] = False
            turn_direction_info['right'][turn_id] = False
            turn_direction_info[''][turn_id] = False
            v1_1 = np.array(agent_front[turn_id])
            v2_1 = np.array(object_pos1)- np.array(agent_pos[turn_id])
            v1_2 = np.array(agent_front[turn_id])
            v2_2 = np.array(object_pos2)- np.array(agent_pos[turn_id])
            d_1 = direction_judging(v1_1,v2_1,mode='L&R')
            d_2 = direction_judging(v1_2,v2_2,mode='L&R')
            if (d_1==d_2):
                turn_direction_info[d_1][turn_id]=True  
        info = {object_name1:object_caption1, object_name2:object_caption2}
        if sum([turn_direction_info['left'][turn_id] for turn_id in turn_direction_info['left']])==1:
            for turn_id in turn_direction_info['left']:
                if not turn_direction_info['left'][turn_id]:
                    continue
                question = f'At the end of which turn were both of "{object_caption1}" and "{object_caption2}" on your left side?'
                answer = turn_id+ TURN_BIAS
                sample_pools.append((question,answer,info))
        if sum([turn_direction_info['right'][turn_id] for turn_id in turn_direction_info['right']])==1:
            for turn_id in turn_direction_info['right']:
                if not turn_direction_info['right'][turn_id]:
                    continue
                question = f'At the end of which turn were both of "{object_caption1}" and "{object_caption2}" on your right side?'
                answer = turn_id+ TURN_BIAS
                sample_pools.append((question,answer,info))
             
    return sample_pools

# TYPE-2: Agent-pos/info

def generate_for_A_existence_option(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_AG_existence1: 基于全局信息确定是否有某个类别
    '''
    sample_pools = []
    positive_types = []
    
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in positive_types and object_type in common_type:
            positive_types.append(object_type)
    negative_types = [type_name for type_name in common_type if type_name not in EXCULED_OBJECT_TYPE and type_name not in positive_types]    
    
    negative_types = random.sample(negative_types,min(len(positive_types),len(negative_types)))
    info = {'pos_type':positive_types}
    for object_type in negative_types:    
        question = f'Remember, have you seen any {object_type}(s) so far? '
        answer = f'No' 
        sample_pools.append((question,answer,['Yes','No'],info))
    for object_type in positive_types:    
        question = f'Remember, have you seen any {object_type}(s) so far? '
        answer = f'Yes' 
        sample_pools.append((question,answer,['Yes','No'],info))
     
    return sample_pools
def generate_for_A_existence_int1(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_TG_appearance1: 什么时候第一次见某个物体，对于首轮问题不成立
    '''
    sample_pools = []
    object_pools = {object_name:current_total_visible[object_name].get("caption",f"the {object_name.split('_')[0]}").replace("a ","the ") for object_name in current_total_visible if  object_name.split('_')[0] in unqine_names}

    info = {'pos_object':object_pools}
    if current_turn_id==0:
        return []
    for object_name in current_total_visible.keys():
        object_type = object_name.split('_')[0]
        if object_type not in unqine_names:
            continue
        skip_flag = False
            
        if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
            skip_flag = True
        if skip_flag:
            continue  
        object_caption = current_total_visible[object_name].get("caption",f"the {object_type}").replace("a ","the ")
        question = f'When did you first discover \"{object_caption}\"? (index of the turn)'
        answer = -1
        for turn_id in range(current_turn_id+1):
            if object_name in current_turn_info[turn_id]["visible_object_info"].keys():
                answer = turn_id+TURN_BIAS 
                break
        assert answer>=0
        info = {object_name:object_caption}
        sample_pools.append((question,answer,info))

    return sample_pools
def generate_for_A_existence_int2(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_TG_appearance3: 见到的特殊性事件
    '''
    sample_pools = []
    object_pools = {object_name:current_total_visible[object_name].get("caption",f"the {object_name.split('_')[0]}").replace("a ","the ") for object_name in current_total_visible if  object_name.split('_')[0] in unqine_names}

   
    if current_turn_id==0:
        return []
    unqine_object_list_dict = {}
    unqine_object_list_total = []
    for turn_id in range(current_turn_id+1):
        unqine_object_list_dict[turn_id] = []
        for object_name in current_turn_info[turn_id]["visible_object_info"].keys():
            object_type = object_name.split('_')[0]
            if object_type not in unqine_names:
                continue
            if object_name not in unqine_object_list_total:
                unqine_object_list_total.append(object_name)

        raw_tuple_set1 = list(itertools.combinations(unqine_object_list_total, 2))
        raw_tuple_set2 = list(itertools.combinations(unqine_object_list_total, 3))
        for tuple_ in raw_tuple_set1:
            skip_flag = False
            for object_name in tuple_:
                object_type = object_name.split('_')[0]
                if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                    skip_flag = True
            if skip_flag:
                continue
            cnt = 0
            target_turn_id = -1
            for turn_id in range(current_turn_id+1):
                if  all(item_ in current_turn_info[turn_id]["visible_object_info"].keys() for item_ in tuple_):
                    cnt+=1
                    target_turn_id = turn_id
            if cnt==1:
                
                object_type1 = tuple_[0].split('_')[0] 
                object_type2 = tuple_[1].split('_')[0] 
                object_caption1 = current_total_visible[tuple_[0]].get("caption",f"the {object_type1}").replace("a ","the ")
                object_caption2 = current_total_visible[tuple_[1]].get("caption",f"the {object_type2}").replace("a ","the ")
                question = f'In which turn did you see both \"{object_caption1}\" and \"{object_caption2}\" simultaneously? '
                answer = target_turn_id+TURN_BIAS 
                info = {tuple_[0]:object_caption1, tuple_[1]:object_caption2}
                sample_pools.append((question,answer,info))

    return sample_pools
def generate_for_A_quantity(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_AG_quantity: 基于全局信息确定物体个数, 排除是1的
    '''
    sample_pools = []
    positive_types = []
    num_cnt= {}
    for object_name in current_total_visible:
        object_type,object_id = object_name.split("_")
        if object_type not in positive_types:
            num_cnt[object_type] = 0
            positive_types.append(object_type)
        num_cnt[object_type] +=1
    positive_types = [type_name for type_name in positive_types if num_cnt[type_name]>1 and type_name not in not_countable and type_name in common_type]
    negative_types = [type_name for type_name in common_type if type_name not in EXCULED_OBJECT_TYPE and type_name not in positive_types and type_name not in not_countable]    
    
    negative_types = random.sample(negative_types,min(int(0.5*len(positive_types)),len(negative_types)))
    
    for object_type in negative_types:    
        question = f'Remember, how many {object_type}(s) have you seen so far? '
        answer = 0 
        sample_pools.append((question,answer))
    for object_type in positive_types:    
        question = f'Remember, how many {object_type}(s) have you seen so far? '
        answer = num_cnt[object_type]
        

        sample_pools.append((question,answer))
     
    return sample_pools
def generate_for_A_diversity(disappear_objects,new_appear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_AG_quantity: 对比新一轮可见与不可见性，第一轮不在考虑范围内
    '''
    sample_pools = []
    object_pools = {object_name:current_total_visible[object_name].get("caption",f"the {object_name.split('_')[0]}").replace("a ","the ") for object_name in current_total_visible if  object_name.split('_')[0] in unqine_names}

    
    
    if current_turn_id==0:
        return []
    current_objects = set(current_total_visible.keys()) - disappear_objects
    
    # 新发现了什么物体
    negative_objects = current_objects - new_appear_objects
    positive_objects = new_appear_objects
    negative_objects = [object_name for object_name in negative_objects if object_name.split('_')[0] in unqine_names]
    positive_objects = [object_name for object_name in positive_objects if object_name.split('_')[0] in unqine_names]
    negative_tuple_set = list(itertools.combinations(negative_objects , 2))
    for (negative_object1, negative_object2) in negative_tuple_set:
        for positive_object3 in positive_objects:
            negative_object1_caption = current_total_visible[negative_object1].get("caption",f"the {negative_object1.split('_')[0]}").replace("a ","the ")
            negative_object2_caption = current_total_visible[negative_object2].get("caption",f"the {negative_object2.split('_')[0]}").replace("a ","the ")
            positive_object3_caption = current_total_visible[positive_object3].get("caption",f"the {positive_object3.split('_')[0]}").replace("a ","the ")
            skip_flag = False
            for object_name in [negative_object1,negative_object2,positive_object3]:
                object_type = object_name.split('_')[0]
                if current_total_visible[object_name].get("caption",f"the {object_type}")==f"the {object_type}" and object_type in not_refer:
                    skip_flag = True
            if skip_flag:
                continue
            caption_list = [negative_object1_caption,negative_object2_caption,positive_object3_caption]
            random.shuffle(caption_list)
            question = f'Among these three objects, which one was newly discovered in this turn(had not appeared before)? \"{caption_list[0]}\"; \"{caption_list[1]}\"; \"{caption_list[2]}\" '
            answer = positive_object3_caption 
            option = [caption_list[0],caption_list[1],caption_list[2]]
            info = {negative_object1:negative_object1_caption, negative_object2:negative_object2_caption,positive_object3:positive_object3_caption }
            sample_pools.append((question,answer,option,info))
    return sample_pools
def generate_for_A_area(obj_pcds,countable_list):
    import alphashape
    countable_pcd_list = [obj_pcds[int(object_name.split("_")[1])] for object_name in countable_list if int(object_name.split("_")[1]) in obj_pcds]
    if len(countable_pcd_list)<3:
        return []
    all_point_xy = np.concatenate([pcd[:,:2] for pcd in countable_pcd_list],axis=0)
    visible_room_size = alphashape.alphashape(all_point_xy, 0.0).area
    question = f'How much space has your field of view traversed so far? (in square meters)'
    answer = item_rounding(visible_room_size)
    return [(question,answer)]*3
def generate_for_A_order(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id):
    '''template for no_TG_appearance2: 见到物体的类别顺序
    '''
    
    
    def exist_repeat(item_dict):
        cnt_dict = {}
        for key_ in item_dict:
            if item_dict[key_] not in cnt_dict:
                cnt_dict[item_dict[key_]] = 1
            else:
                return True
        return False
    sample_pools = []
    appearance_info = {}
    if current_turn_id==0:
        return []
    for turn_id in range(current_turn_id+1):
        for object_name in current_turn_info[turn_id]["visible_object_info"].keys():
            object_type = object_name.split('_')[0]
            
            if object_type not in appearance_info and object_type in common_type:
                appearance_info[object_type] = turn_id           
    
    if len(appearance_info.keys())<4:
        return []           
    raw_tuple_set = list(itertools.combinations(list(appearance_info.keys()), 3))
    info = {'pos_type': list(appearance_info.keys())}
    for tuple_list in raw_tuple_set:
        if exist_repeat({k:v for (k,v) in appearance_info.items() if k in tuple_list}):
            continue
        order_dict = {k:v for (k,v) in appearance_info.items() if k in tuple_list}
        sorted_list = sorted(order_dict.items(), key=lambda item: item[1])
        order_dict = {k:v for (k,v) in sorted_list}
        
        gt_tuple_order = list(order_dict.keys())
       
        show_tuple_order = list(tuple_list)
        random.shuffle(show_tuple_order)
        question = f'What will be the first-time appearance order of the following categories: {show_tuple_order[0]}, {show_tuple_order[1]}, {show_tuple_order[2]} ? '
        answer = f'{gt_tuple_order[0]}, {gt_tuple_order[1]}, {gt_tuple_order[2]}'
        assert appearance_info[gt_tuple_order[0]]<appearance_info[gt_tuple_order[1]], print({k:v for (k,v) in appearance_info.items() if k in tuple_list})
        all_orders = permutations(show_tuple_order)
        all_possible_text = [f'{item[0]}, {item[1]}, {item[2]}' for item in all_orders]
        all_possible_text = [text for text in all_possible_text if text!=answer]
        options = random.sample(all_possible_text,3)
        options.append(answer)
        shuffled_options = random.sample(options,len(options))
        sample_pools.append((question,answer,shuffled_options,info))

    return sample_pools
def generate_for_A_spatial(agent_pos,agent_front,current_turn_id):
    # 向前还是向后，向左还是向右，左转还是右转
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product == 0:
            cross_product = -1
        raw_angle = np.arccos(cos_theta)*180/np.pi * -np.sign(cross_product)
        if raw_angle < 0:
            raw_angle+=360.0
        return raw_angle 
    agent_dis_threshold = 1.0
    agent_angle_threshold = 20
    sample_distance_option = []
    sample_distance_float = []
    sample_direction_option = []
    sample_direction_float = []
    
    for past_turn_id in range(current_turn_id):
        agent_pos_1 = np.array(agent_pos[past_turn_id])
        agent_pos_2 = np.array(agent_pos[current_turn_id])
        agent_front_1 = np.array(agent_front[past_turn_id])
        agent_front_2 = np.array(agent_front[current_turn_id])
        vector = agent_pos_2 - agent_pos_1
        angle = get_angle(agent_front_1,vector)
        vector_forward = np.linalg.norm(vector)*math.cos(math.radians(angle))
        vector_right = np.linalg.norm(vector)*math.sin(math.radians(angle))
        
        if abs(vector_right)>agent_dis_threshold:
            question_1 = f'Assuming the direction you are facing at the end of the turn {past_turn_id+TURN_BIAS} is forward, did you move a certain distance left or right from that position?'
            option_1 = ['Left','Right']
            answer_1 = 'Right' if vector_right>0 else 'Left'
            sample_distance_option.append((question_1,answer_1,option_1))
        if abs(vector_forward)>agent_dis_threshold:
            question_2 = f'Assuming the direction you are facing at the end of the turn {past_turn_id+TURN_BIAS} is forward, did you move a certain distance forward or backward from that position?'
            option_2 = ['Forward','Backward']
            answer_2 = 'Forward' if vector_forward>0 else 'Backward'
            sample_distance_option.append((question_2,answer_2,option_2))
        if np.linalg.norm(vector)>agent_dis_threshold:
            question_3 = f'How far is your current position from where you were at the end of the turn {past_turn_id+TURN_BIAS}? (in meters)'
            answer_3 = item_rounding(np.linalg.norm(vector))
            sample_distance_float.append((question_3,answer_3))
        
        
        front_change = get_angle(agent_front_1,agent_front_2)
        if (front_change>agent_angle_threshold and front_change<180-agent_angle_threshold) or (front_change>180+agent_angle_threshold and front_change<360-agent_angle_threshold):
            question_1 = f'Using your orientation at the end of turn {past_turn_id+TURN_BIAS} as a reference, has your current orientation rotated clockwise or counterclockwise by a certain angle (<180) relative to that orientation?'
            option_1 = ['Clockwise','Counterclockwise']
            answer_1 = 'Clockwise' if front_change<180 else 'Counterclockwise'
            sample_direction_option.append((question_1,answer_1,option_1))
            question_2 = f'Using your orientation at the end of turn {past_turn_id+TURN_BIAS} as a reference, by how many degrees has your current orientation rotated {answer_1.lower()} relative to that previous orientation?'
            answer_2 = item_rounding(front_change) if front_change<180 else item_rounding(360-front_change)
            sample_direction_float.append((question_2,answer_2))
            
            
            
                
    return sample_distance_option,sample_distance_float,sample_direction_option,sample_direction_float


def generate_for_each_turn(info_of_turns, current_turn_id, QA_Sizer):
    
    # all history info the agent can obtain
    current_turn_info = {k:v for k,v in info_of_turns.turn_info.items() if k <= current_turn_id}
    current_total_visible = info_of_turns.prefix_visible[current_turn_id]
    
    # info of agent, the ending state of the agent in each turn 
    agent_pos = {turn_id: current_turn_info[turn_id]["position"][-1] for turn_id in current_turn_info.keys()}
    agent_pos_wz = {turn_id: current_turn_info[turn_id]["position_wz"][-1] for turn_id in current_turn_info.keys()}
    agent_front = {turn_id: current_turn_info[turn_id]["orientation"][-1] for turn_id in current_turn_info.keys()}
    
    # preprocess some useful info for template generation
    disappear_objects = set(current_total_visible.keys()) - set(current_turn_info[current_turn_id]["visible_object_info"].keys())
    new_appear_objects = set(current_turn_info[current_turn_id]["visible_object_info"].keys()) - set(info_of_turns.prefix_visible[current_turn_id-1].keys()) \
        if current_turn_id>0 else set(current_turn_info[current_turn_id]["visible_object_info"].keys()) 
    object_type_dict = {}
    for object_name in current_total_visible:
        object_type, object_id = object_name.split("_")
        object_id = int(object_id)
        if object_type not in object_type_dict:
            object_type_dict[object_type] = []
        object_type_dict[object_type].append(object_id)
    unqine_names = [object_type for object_type in object_type_dict.keys() if len(object_type_dict[object_type]) == 1]
    
    countable_list = current_turn_info[current_turn_id]["countable_info"]
    obj_pcds = info_of_turns.obj_pcds
    sample_pools = {}
    
    # Agent-object spatial:
  
    sample_pools['AO_direction(option1)'] = generate_for_AO_direction_option_2d(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)
    sample_pools['AO_direction(option2)'] = generate_for_AO_direction_option_4d(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)
    sample_pools['AO_direction(option3)'] = generate_for_AO_direction_option_3o(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)
    
    sample_pools['AO_distance(option1)'] = generate_for_AO_distance_option_2a(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list)
    sample_pools['AO_distance(option2)'] = generate_for_AO_distance_option_3o(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list)
    sample_pools['AO_distance(option3)'] = generate_for_AO_distance_option_2o2a(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos_wz,agent_front,current_turn_id,countable_list)
    
    sample_pools['AO_distance(int)'] = generate_for_AO_distance_int(obj_pcds,current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)
    sample_pools['AO_direction(int)'] = generate_for_AO_direction_int(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)

    sample_pools['AO_distance(float)'] = generate_for_AO_distance_float(obj_pcds,disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)
    sample_pools['AO_direction(float)'] = generate_for_AO_direction_float(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id,countable_list)

    # template type 2:
    
    # Temporal grounding relating to visiblity
    sample_pools['A_object-order(option)'] = generate_for_A_order(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_object-existence(option)'] = generate_for_A_existence_option(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_object-existence(int1)'] = generate_for_A_existence_int1(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_object-existence(int2)']  = generate_for_A_existence_int2(current_turn_info,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_object-quantity(int)'] = generate_for_A_quantity(disappear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_object-diversity(option)'] = generate_for_A_diversity(disappear_objects,new_appear_objects,unqine_names,current_total_visible,agent_pos,agent_front,current_turn_id)
    sample_pools['A_room-size(float)'] = generate_for_A_area(obj_pcds,countable_list)
    
    # Agent pose
    sample_pools['A_position(option)'],sample_pools['A_position(float)'],sample_pools['A_orientation(option)'],sample_pools['A_orientation(float)'] = generate_for_A_spatial(agent_pos,agent_front,current_turn_id)
    
  
    type_,sample_,candidate_ = QA_Sizer.choose_from_samples(sample_pools)
    new_candidate_ = []
    for item_tuple in candidate_:
        new_item_dict = {
            'question':item_tuple[0],
            'answer':item_tuple[1],
            'option':[],
            'info':{}}
        if len(item_tuple)==2:
            pass
        elif len(item_tuple)==3:
            if isinstance(item_tuple[2],list):
                new_item_dict['option'] = item_tuple[2]
            else:
                new_item_dict['info'] = item_tuple[2]
        else:
            new_item_dict['option'] = item_tuple[2]
            new_item_dict['info'] = item_tuple[3]
        new_candidate_.append(new_item_dict)
    
     
    sample_dict = {'turn_id':current_turn_id+TURN_BIAS,
                   'type': type_,
                   'image_paths':[current_turn_info[current_turn_id]['images']],
                   'agent_pos':[item_rounding(t) for t in current_turn_info[current_turn_id]["position"]],
                   'agent_front':[item_rounding(t) for t in current_turn_info[current_turn_id]["orientation"]] ,
                   'candidate_':new_candidate_
                    }
    

    return sample_dict    
    
def process_one_scan(scan_id):
    for iter_ in range(DIVERSITY_EACH_SCAN):
        results = []
        qa_sizer = QA_Sizer(WEIGHT_DICT)
        info_of_turns = Info_of_turns(scan_id)
        for current_turn_id in range(info_of_turns.total_turns):
            results.append(generate_for_each_turn(info_of_turns, current_turn_id,qa_sizer))
        with open(f'{TARGET_DIR}/{scan_id}_{iter_}.json','w') as f:
            json.dump(results,f,indent=4)


def process_one_scan_test(scan_id,test_type,WEIGHT_DICT):        
    results = []
    WEIGHT_DICT = {k:0 for k in WEIGHT_DICT.keys()}
    WEIGHT_DICT[test_type] = 1.0
    qa_sizer = QA_Sizer(WEIGHT_DICT)
    info_of_turns = Info_of_turns(scan_id)
    for current_turn_id in range(info_of_turns.total_turns):
        print(current_turn_id)
        results.append(generate_for_each_turn(info_of_turns, current_turn_id,qa_sizer))
        print(results[-1]['question'],results[-1]['answer']) 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Step1')
    
    parser.add_argument('--split', type=str,default='scannet')
    args = parser.parse_args()
    
    scan_split = args.split
    DATA_DIR =f'./data/process_data/{scan_split}'
    TARGET_DIR = f'./data/step_1/{scan_split}'
    os.makedirs(TARGET_DIR,exist_ok=True)
    
    PCD_DIR = './data/process_pcd/'
    all_task = ['_'.join(scan_name.split('_')[:-1]) for scan_name in os.listdir(DATA_DIR)]
    
    mmengine.utils.track_parallel_progress(process_one_scan, all_task, 12)
    