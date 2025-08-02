import numpy as np
import os
import trimesh
from pytorch3d.transforms import euler_angles_to_matrix
import pickle
import torch
import mmengine
from utils.data_io import raw_info

TYPE2INT = {'adhesive tape': 1, 'air conditioner': 2, 'alarm': 3, 'album': 4, 'arch': 5, 'backpack': 6, 'bag': 7, 'balcony': 8, 'ball': 9, 'banister': 10, 'bar': 11, 'barricade': 12, 'baseboard': 13, 'basin': 14, 'basket': 15, 'bathtub': 16, 'beam': 17, 'beanbag': 18, 'bed': 19, 'bench': 20, 'bicycle': 21, 'bidet': 22, 'bin': 23, 'blackboard': 24, 'blanket': 25, 'blinds': 26, 'board': 27, 'body loofah': 28, 'book': 29, 'boots': 30, 'bottle': 31, 'bowl': 32, 'box': 33, 'bread': 34, 'broom': 35, 'brush': 36, 'bucket': 37, 'cabinet': 38, 'calendar': 39, 'camera': 40, 'can': 41, 'candle': 42, 'candlestick': 43, 'cap': 44, 'car': 45, 'carpet': 46, 'cart': 47, 'case': 48, 'ceiling': 49, 'chair': 50, 'chandelier': 51, 'cleanser': 52, 'clock': 53, 'clothes': 54, 'clothes dryer': 55, 'coat hanger': 56, 'coffee maker': 57, 'coil': 58, 'column': 59, 'commode': 60, 'computer': 61, 'conducting wire': 62, 'container': 63, 'control': 64, 'copier': 65, 'cosmetics': 66, 'couch': 67, 'counter': 68, 'countertop': 69, 'crate': 70, 'crib': 71, 'cube': 72, 'cup': 73, 'curtain': 74, 'cushion': 75, 'decoration': 76, 'desk': 77, 'detergent': 78, 'device': 79, 'dish rack': 80, 'dishwasher': 81, 'dispenser': 82, 'divider': 83, 'door': 84, 'door knob': 85, 'doorframe': 86, 'doorway': 87, 'drawer': 88, 'dress': 89, 'dresser': 90, 'drum': 91, 'duct': 92, 'dumbbell': 93, 'dustpan': 94, 'dvd': 95, 'eraser': 96, 'excercise equipment': 97, 'fan': 98, 'faucet': 99, 'fence': 100, 'file': 101, 'fire extinguisher': 102, 'fireplace': 103, 'floor': 104, 'flowerpot': 105, 'flush': 106, 'folder': 107, 'food': 108, 'footstool': 109, 'frame': 110, 'fruit': 111, 'furniture': 112, 'garage door': 113, 'garbage': 114, 'glass': 115, 'globe': 116, 'glove': 117, 'grab bar': 118, 'grass': 119, 'guitar': 120, 'hair dryer': 121, 'hamper': 122, 'handle': 123, 'hanger': 124, 'hat': 125, 'headboard': 126, 'headphones': 127, 'heater': 128, 'helmets': 129, 'holder': 130, 'hook': 131, 'humidifier': 132, 'ironware': 133, 'jacket': 134, 'jalousie': 135, 'jar': 136, 'kettle': 137, 'keyboard': 138, 'kitchen island': 139, 'kitchenware': 140, 'knife': 141, 'label': 142, 'ladder': 143, 'lamp': 144, 'laptop': 145, 'ledge': 146, 'letter': 147, 'light': 148, 'luggage': 149, 'machine': 150, 'magazine': 151, 'mailbox': 152, 'map': 153, 'mask': 154, 'mat': 155, 'mattress': 156, 'menu': 157, 'microwave': 158, 'mirror': 159, 'molding': 160, 'monitor': 161, 'mop': 162, 'mouse': 163, 'napkins': 164, 'notebook': 165, 'object': 166, 'ottoman': 167, 'oven': 168, 'pack': 169, 'package': 170, 'pad': 171, 'pan': 172, 'panel': 173, 'paper': 174, 'paper cutter': 175, 'partition': 176, 'pedestal': 177, 'pen': 178, 'person': 179, 'piano': 180, 'picture': 181, 'pillar': 182, 'pillow': 183, 'pipe': 184, 'pitcher': 185, 'plant': 186, 'plate': 187, 'player': 188, 'plug': 189, 'plunger': 190, 'pool': 191, 'pool table': 192, 'poster': 193, 'pot': 194, 'price tag': 195, 'printer': 196, 'projector': 197, 'purse': 198, 'rack': 199, 'radiator': 200, 'radio': 201, 'rail': 202, 'range hood': 203, 'refrigerator': 204, 'remote control': 205, 'ridge': 206, 'rod': 207, 'roll': 208, 'roof': 209, 'rope': 210, 'sack': 211, 'salt': 212, 'scale': 213, 'scissors': 214, 'screen': 215, 'seasoning': 216, 'shampoo': 217, 'sheet': 218, 'shelf': 219, 'shirt': 220, 'shoe': 221, 'shovel': 222, 'shower': 223, 'sign': 224, 'sink': 225, 'soap': 226, 'soap dish': 227, 'soap dispenser': 228, 'socket': 229, 'speaker': 230, 'sponge': 231, 'spoon': 232, 'stairs': 233, 'stall': 234, 'stand': 235, 'stapler': 236, 'statue': 237, 'steps': 238, 'stick': 239, 'stool': 240, 'stopcock': 241, 'stove': 242, 'structure': 243, 'sunglasses': 244, 'support': 245, 'switch': 246, 'table': 247, 'tablet': 248, 'teapot': 249, 'telephone': 250, 'thermostat': 251, 'tissue': 252, 'tissue box': 253, 'toaster': 254, 'toilet': 255, 'toilet paper': 256, 'toiletry': 257, 'tool': 258, 'toothbrush': 259, 'toothpaste': 260, 'towel': 261, 'toy': 262, 'tray': 263, 'treadmill': 264, 'trophy': 265, 'tube': 266, 'tv': 267, 'umbrella': 268, 'urn': 269, 'utensil': 270, 'vacuum cleaner': 271, 'vanity': 272, 'vase': 273, 'vent': 274, 'ventilation': 275, 'wall': 276, 'wardrobe': 277, 'washbasin': 278, 'washing machine': 279, 'water cooler': 280, 'water heater': 281, 'window': 282, 'window frame': 283, 'windowsill': 284, 'wine': 285, 'wire': 286, 'wood': 287, 'wrap': 288}



def is_inside_box(points, center, size, rotation_mat):
    """Check if points are inside a 3D bounding box.

    Args:
        points: 3D points, numpy array of shape (n, 3).
        center: center of the box, numpy array of shape (3, ).
        size: size of the box, numpy array of shape (3, ).
        rotation_mat: rotation matrix of the box, numpy array of shape (3, 3).
    Returns:
        Boolean array of shape (n, ) indicating if each point is inside the box.
    """
    assert points.shape[1] == 3, 'points should be of shape (n, 3)'
    center = np.array(center)  # n, 3
    size = np.array(size)  # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (
        3,
        3,
    ), f'R should be shape (3,3), but got {rotation_mat.shape}'
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat  # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return ((pcd_local[:, 0] <= 1)
            & (pcd_local[:, 1] <= 1)
            & (pcd_local[:, 2] <= 1))

def process_arkit(scan_name,align_matrix):
    split,scan_id = scan_name.split('_')[0],scan_name.split('_')[1].split('.')[0]
    plydata = trimesh.load(f'./data/arkitscenes/{split}/{scan_id}/{scan_id}_3dod_mesh.ply', process=False)
    pc = plydata.vertices
    vertex_colors = plydata.visual.vertex_colors
    vertex_colors = vertex_colors[:, :3]
    pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
    pts[:, :3] = pc
    pc = np.dot(pts, align_matrix.transpose())[:, :3].astype(np.float32)
    return pc, vertex_colors
def create_scene_pcd(scan_name, es_anno):
    """Adding the embodiedscan-box annotation into the point clouds data.

    Args:
        es_anno (dict): The embodiedscan annotation of
            the target scan.
        pcd_result (tuple) :
            (1) aliged point clouds coordinates
                shape (n,3)
            (2) point clouds color ([0,1])
                shape (n,3)
            (3) label (no need here)

    Returns:
        tuple :
            (1) aliged point clouds coordinates
                shape (n,3)
            (2) point clouds color ([0,1])
                shape (n,3)
            (3) point clouds label (int)
                shape (n,1)
            (4) point clouds object id (int)
                shape (n,1)
    """
    pc, color = process_arkit(scan_name,es_anno["axis_align_matrix"])
    label = np.ones_like(color[:,0]) * -100
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    bboxes = es_anno['bboxes'].reshape(-1, 9)
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    object_ids = es_anno['object_ids']
    object_types = es_anno['object_types']  # str
    sorted_indices = sorted(enumerate(bboxes),
                            key=lambda x: -np.prod(x[1][3:6]))
    # the larger the box, the smaller the index
    sorted_indices_list = [index for index, value in sorted_indices]

    bboxes = [bboxes[index] for index in sorted_indices_list]
    object_ids = [object_ids[index] for index in sorted_indices_list]
    object_types = [object_types[index] for index in sorted_indices_list]
    #print(object_ids)
    for box, obj_id, obj_type in zip(bboxes, object_ids, object_types):
        
        obj_type_id = TYPE2INT.get(obj_type, -1)
        center, size = box[:3], box[3:6]

        orientation = np.array(
            euler_angles_to_matrix(torch.tensor(box[np.newaxis, 6:]),
                                   convention='ZXY')[0])

        box_pc_mask = is_inside_box(pc, center, size, orientation)

        instance_ids[box_pc_mask] = obj_id
   
        label[box_pc_mask] = obj_type_id
    return pc, color, label, instance_ids

def process_one_scan(scan_name):
    scene_info = raw_info[scan_name]
    
    save_path = f'./data/process_pcd/{scan_name.split(".")[0]}.pth'
    if os.path.exists(save_path):
        return
    torch.save(create_scene_pcd(scan_name, scene_info), save_path)
    try:
        torch.save(create_scene_pcd(scan_name, scene_info), save_path)
    except:
        print(f'Error in processing {scan_name}')
if __name__ == '__main__':

    all_arkit_scene = [scan_id for scan_id in raw_info.keys() if 'Training' in scan_id or 'Validation' in scan_id]
    mmengine.utils.track_parallel_progress(process_one_scan, all_arkit_scene, 12)

  