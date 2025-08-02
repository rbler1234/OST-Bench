import numpy as np
from tqdm import tqdm
import json

global raw_info 
global scene_info
raw_info= {}
scene_info = {}
with open('./meta_data/mp3d_mapping.json') as f:
    mp3d_json = json.load(f)
    

    

def rename_scan(scan_name):
    if 'arkitscenes' in scan_name:
        return scan_name.split('/')[1]+'_'+scan_name.split('/')[2]
    elif 'matterport3d' in scan_name:
        parts = scan_name.split('/')
        
        return f"{mp3d_json[parts[1]]}_{parts[2]}"
    else:
        return scan_name.split('/')[-1]

def read_annotation_pickle(path: str, show_progress: bool = True):
    """Read annotation pickle file and return a dictionary, the embodiedscan
    annotation for all scans in the split.

    Args:
        path (str): the path of the annotation pickle file.
        show_progress (bool): whether showing the progress.
    Returns:
        dict: A dictionary.
            scene_id : (bboxes, object_ids, object_types,
                visible_view_object_dict, extrinsics_c2w,
                axis_align_matrix, intrinsics, image_paths)
            bboxes:
                numpy array of bounding boxes,
                shape (N, 9): xyz, lwh, ypr
            object_ids:
                numpy array of obj ids, shape (N,)
            object_types:
                list of strings, each string is a type of object
            visible_view_object_dict:
                a dictionary {view_id: visible_instance_ids}
            extrinsics_c2w:
                a list of 4x4 matrices, each matrix is the extrinsic
                matrix of a view
            axis_align_matrix:
                a 4x4 matrix, the axis-aligned matrix of the scene
            intrinsics:
                a list of 4x4 matrices, each matrix is the intrinsic
                matrix of a view
            image_paths:
                a list of strings, each string is the path of an image
                in the scene
    """
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)

    metainfo = data['metainfo']
    object_type_to_int = metainfo['categories']
    object_int_to_type = {v: k for k, v in object_type_to_int.items()}
    datalist = data['data_list']
    output_data = {}
    pbar = (tqdm(range(len(datalist))) if show_progress else range(
        len(datalist)))
    for scene_idx in pbar:

        images = datalist[scene_idx]['images']

        intrinsic = datalist[scene_idx].get('cam2img', None)  # a 4x4 matrix
        missing_intrinsic = False
        if intrinsic is None:
            missing_intrinsic = (
                True  # each view has different intrinsic for mp3d
            )
        depth_intrinsic = datalist[scene_idx].get(
            'cam2depth', None)  # a 4x4 matrix, for 3rscan
        if depth_intrinsic is None and not missing_intrinsic:
            depth_intrinsic = datalist[scene_idx][
                'depth_cam2img']  # a 4x4 matrix, for scannet
        axis_align_matrix = datalist[scene_idx][
            'axis_align_matrix']  # a 4x4 matrix

        scene_id = datalist[scene_idx]['sample_idx']
        if 'instances' in datalist[scene_idx]:
            instances = datalist[scene_idx]['instances']
            bboxes = []
            object_ids = []
            object_types = []
            object_type_ints = []
            for object_idx in range(len(instances)):
                bbox_3d = instances[object_idx]['bbox_3d']  # list of 9 values
                bbox_label_3d = instances[object_idx]['bbox_label_3d']  # int
                bbox_id = instances[object_idx]['bbox_id']  # int
                object_type = object_int_to_type[bbox_label_3d]

                object_type_ints.append(bbox_label_3d)
                object_types.append(object_type)
                bboxes.append(bbox_3d)
                object_ids.append(bbox_id)
            bboxes = np.array(bboxes)
            object_ids = np.array(object_ids)
            object_type_ints = np.array(object_type_ints)

        visible_view_object_dict = {}
        extrinsics_c2w = []
        intrinsics = []
        depth_intrinsics = []
        depth_image_paths = []
        image_paths = []
        visible_list = []
        for image_idx in range(len(images)):
            img_path = images[image_idx]["img_path"]  # str
            depth_path = images[image_idx]["depth_path"]
            extrinsic_id = img_path.split("/")[-1].split(".")[0]  # str
            cam2global = images[image_idx]["cam2global"]  # a 4x4 matrix
            if missing_intrinsic:
                intrinsic = images[image_idx]["cam2img"]
                depth_intrinsic = images[image_idx]["cam2img"]
            visible_instance_indices = images[image_idx][
                "visible_instance_ids"
            ]  # numpy array of int
            visible_list.append(visible_instance_indices)
            visible_instance_ids = object_ids[visible_instance_indices]
            visible_view_object_dict[extrinsic_id] = visible_instance_ids
            extrinsics_c2w.append(cam2global)
            intrinsics.append(intrinsic)
            depth_intrinsics.append(depth_intrinsic)
            image_paths.append(img_path)
            depth_image_paths.append(depth_path)
        if show_progress:
            pbar.set_description(f"Processing scene {scene_id}")
        output_data[rename_scan(scene_id)] = {
            "bboxes": bboxes,
            "object_ids": object_ids,
            "object_types": object_types,
            "object_type_ints": object_type_ints,
            "visible_list": visible_list,
            "extrinsics_c2w": extrinsics_c2w,
            "axis_align_matrix": axis_align_matrix,
            "intrinsics": intrinsics,
            "depth_intrinsics": depth_intrinsics,
            "image_paths": image_paths,
            "depth_image_paths":depth_image_paths
        }
    return output_data

raw_info = read_annotation_pickle('./data/embodiedscan-v2/embodiedscan_infos_val.pkl')


def get_scene_info(scene_id):
    global scene_info, render_info, output_dir, painted_dir,raw_info
    if scene_id in scene_info:
        return scene_info[scene_id]
    else:
        anno = raw_info[scene_id]
        if anno is None:
            return None
        scene_info[scene_id] = {}
        scene_info[scene_id]["bboxes"] = anno["bboxes"]
        scene_info[scene_id]["object_ids"] = anno["object_ids"]
        scene_info[scene_id]["object_types"] = anno["object_types"]
        scene_info[scene_id]["visible_list"] = anno["visible_list"]
        scene_info[scene_id]["image_paths"] = anno["image_paths"]
        scene_info[scene_id]["depth_image_paths"] = anno["depth_image_paths"]
        scene_info[scene_id]["depth_intrinsics"] = anno["depth_intrinsics"]
        scene_info[scene_id]["intrinsics"] = anno["intrinsics"]
  
        scene_info[scene_id]["view_ids"] = [path.split("/")[-1].split(".")[0] for path in anno["image_paths"]]
        scene_info[scene_id]["extrinsics_c2w"] = anno["extrinsics_c2w"]
        
        scene_info[scene_id]["axis_align_matrix"] = anno["axis_align_matrix"]
        scene_info[scene_id]["camera_extrinsics_c2w"] = [(anno["axis_align_matrix"] @ extrinsic) for extrinsic in
                                                            anno["extrinsics_c2w"]]
    
        return scene_info[scene_id]