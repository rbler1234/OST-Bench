import os
import json
import time
import pickle
import random
from math import pi, ceil
from tqdm import tqdm

import cv2
import numpy as np
import mmengine
from utils.check_visible import count_inside_image_final, update_vertex_visibility
from utils.data_io import get_scene_info,raw_info


# =============== Constants ===============
THRESHOLD_1 = 0.15
THRESHOLD_2 = 0.03
EXCLUDED_TYPES = ['object', 'ceiling', 'wall', 'floor', 'column']

# =============== Path Definitions ===============

INFO_SAVE_DIR = "./data/embodiedscan_info/"
IMAGE_DIR = './data/arkitscenes/'
OBJECT_CAPTION_DIR = './meta_data/arkitscenes.json'
TARGET_DIR = './data/process_data/arkitscenes'
os.makedirs(TARGET_DIR,exist_ok=True)


# =============== Data Loading ===============


# Load object captions
with open(OBJECT_CAPTION_DIR) as f:
    object_caption_full = json.load(f)

# =============== Helper Functions ===============
def scan_id_map(scan_id):
    """Map scan ID to standardized format."""
    parts = scan_id.split('/')
    return f"{mp3d_json[parts[1]]}_{parts[2]}"

# =============== Data Processing ===============
# Process object captions
scene_info = {}
for scan_id in object_caption_full:
    if 'matterport3d' in scan_id:
        object_caption_full[scan_id_map(scan_id)] = object_caption_full[scan_id]


class trajectory_buffer:
    def __init__(self, scan_info):
        self.buffer = []
        self.full_visible = set()
        self.tmp_visible = []
        self.instance_info = {'type':[],
                              'id':[],
                              'coord':[],
                              'box':[]}
        self.instance_info['type'] = scan_info['object_types']
        self.instance_info['id'] = scan_info['object_ids']
        self.instance_info['coord'] = scan_info['bboxes'][:, :2]
        self.instance_info['box'] = scan_info['bboxes']
        
    def update(self, trajectory):
        self.buffer.append(trajectory)
        for frame_info in trajectory:
            self.full_visible = self.full_visible.union(frame_info['visible_indices'])
        self.tmp_visible = trajectory[-1]['visible_indices']
    
    

def turn_the_image(c2w_matrix):
    judge_value = c2w_matrix[2,1]
    if round(judge_value,0) == 1.0:
        return 0
    elif round(judge_value,0) == 0.0:
        return 90
    elif round(judge_value,0) == -1.0:
        return 180
    else:
        print("can not find the direction of the camera")
   
def generate_text_prompt(scene_id,sapmle_id , camera_index_list,num_interval,num_images):
    
    # init the countable objects state:
    object_vertex_info = {}
    object_types = get_scene_info(scene_id)['object_types']
    object_ids = get_scene_info(scene_id)['object_ids']
    for object_type, object_id in zip(object_types, object_ids):
        object_name = object_type+'_'+str(object_id)
        object_vertex_info[object_name] = {'center':False,
                                            'corner':[False]*8
                                            }
    obj_caption_dict = {}
    if scene_id.split('_')[1] in object_caption_full:
        for item in object_caption_full[scene_id.split('_')[1]]:
            obj_caption_dict[item['object_id']] = item['object_caption']
   
    for k in camera_index_list:
        step_id = k
        visible_set = set()
        tmp_pos = None
        trajory_info = []
        c_info = []
        for view_id_idx in range(len(get_scene_info(scene_id)['camera_extrinsics_c2w']))[k*(num_interval*num_images):(k+1)*(num_interval*num_images):num_interval]:
        
            
            extrinsics_c2w = get_scene_info(scene_id)['camera_extrinsics_c2w'][view_id_idx]
            front = extrinsics_c2w[:3, 2]
            #print(extrinsics_c2w[:, 3])
            pos = extrinsics_c2w[:3, 3]
            ex_in_world, ey_in_world = pos[0], pos[1]
    
            c_info.append(tmp_pos)
            
            
            
            # draw obj in the photos
            vis_list = get_scene_info(scene_id)['visible_list'][view_id_idx]
            image_path = get_scene_info(scene_id)['image_paths'][view_id_idx] 
            real_image_path = IMAGE_DIR+image_path.split('arkitscenes/')[1]
            real_depth_path = real_image_path.replace('lowres_wide','lowres_depth')  
            real_image = cv2.imread(real_image_path)
            real_depth = cv2.imread(real_depth_path,cv2.IMREAD_UNCHANGED)
            real_depth = real_depth.astype(np.float32)/1000.0
            extrinsics_c2w = np.matmul(get_scene_info(scene_id)["axis_align_matrix"], get_scene_info(scene_id)["extrinsics_c2w"])
            extrinsic_c2w = extrinsics_c2w[view_id_idx]
            filter_vis_list = []
            for idx in vis_list:
                object_box = get_scene_info(scene_id)['bboxes'][idx]
                object_type = get_scene_info(scene_id)['object_types'][idx]
                object_id = get_scene_info(scene_id)['object_ids'][idx]
                object_name = f"{object_type}_{object_id}"

                
                if object_type in EXCLUDED_TYPES:
                    continue
                
                num_1,num_2,_ = count_inside_image_final(real_depth,object_box,extrinsic_c2w,get_scene_info(scene_id)['depth_intrinsics'][view_id_idx])
                if (num_2/num_1)>=THRESHOLD_1 or num_2/(real_depth.shape[0]*real_depth.shape[1])>=THRESHOLD_2:
                    filter_vis_list.append(idx)
                
                    
                center_state, object_vertex_info[object_name]['corner'] = update_vertex_visibility(real_image,object_box,extrinsic_c2w,
                                                                                        get_scene_info(scene_id)['intrinsics'][view_id_idx],object_vertex_info[object_name]['corner'])
                if center_state:
                    object_vertex_info[object_name]['center'] = True
            trajory_info.append(
                {
                    'pos':(ex_in_world, ey_in_world),
                    'front':(front[0], front[1]),
                    'pos_xyz':(pos[0], pos[1], pos[2]),
                    
                    'image_path':get_scene_info(scene_id)['image_paths'][view_id_idx] ,
                    'turn image': turn_the_image(extrinsic_c2w)
                }
            )
            visible_set.update(filter_vis_list)
        map_info = {}
        for view_index in visible_set:
            bbox = get_scene_info(scene_id)['bboxes'][view_index]
            object_type = get_scene_info(scene_id)['object_types'][view_index]
            object_id = get_scene_info(scene_id)['object_ids'][view_index]
            
            save_text = object_type+'_'+str(object_id)
            c_x,c_y = bbox[0],bbox[1]
            map_info[save_text] = {'pos':(c_x,c_y),'box':list(bbox)}
        
        countable_list = []
        for object_name in object_vertex_info:
            if object_vertex_info[object_name]['center'] and sum(object_vertex_info[object_name]['corner'])>4:
                countable_list.append(object_name)
        
        
        os.makedirs(f"{TARGET_DIR}/{scene_id}_{sapmle_id}",exist_ok=True)
        with open(f"{TARGET_DIR}/{scene_id}_{sapmle_id}/info_{step_id}.json",'w') as f:
            json.dump({"trajory_info":trajory_info,"map_info":map_info,"countable_info":countable_list,"visible_info":list(map_info.keys())},f,indent=4)
def process_scene(scene_id):
    
    camera_index_list = []
    if len(get_scene_info(scene_id)['extrinsics_c2w'])>=160:
        num_interval = 4
        num_images = 5
    elif len(get_scene_info(scene_id)['extrinsics_c2w'])>=100:
        num_interval = 4
        num_images = 4
    else:
        num_interval = 3
        num_images = 4
    for k in range(len(get_scene_info(scene_id)['extrinsics_c2w'])//(num_interval*num_images)):
        camera_index_list.append(k)
        
    generate_text_prompt(scene_id, 0, camera_index_list,num_interval,num_images)
    
    
if __name__ == "__main__":
    tasks = []
    for scan_id_pkl in tqdm(raw_info.keys()):
        if "Training" in scan_id_pkl or 'Validation' in scan_id_pkl:
            tasks.append(scan_id_pkl)
    #process_scene('Training_40776204')    
    mmengine.utils.track_parallel_progress(process_scene, tasks,12)

   
