import numpy as np
from scipy.spatial.distance import cdist

def direction_judging(v1,v2,mode = 'L&R'):
    """Determines the directional relationship between two vectors (e.g., left/right/front/back).

    Based on the angle between two vectors and the specified mode, returns a directional label.
    Supports multiple modes (e.g., left/right, four-way directions).

    Args:
        v1 (array-like): The first vector (2D).
        v2 (array-like): The second vector (2D).
        mode (str, optional): Direction judgment mode. Options:
            - 'L&R': Left/Right only (default)
            - 'L&R&B': Left/Right/Back
            - Other formats trigger four-way (front/rear-left/right) mode.

    Returns:
        str: Direction label. Possible values:
            - 'left', 'right', 'back' (for 'L&R' or 'L&R&B' modes)
            - 'front-right', 'rear-right', 'rear-left', 'front-left' (four-way mode)
            - Empty string if vectors are nearly aligned (within threshold).

    Example:
        >>> direction_judging([1, 0], [0, 1], mode='L&R')
        'right'
        >>> direction_judging([1, 0], [-1, 0], mode='L&R&B')
        'back'
    """

    ANGLE_THRESHOLD = 20
    
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

    if len(mode.split('&'))==3:
        angle = get_angle(v1,v2)
        if angle<ANGLE_THRESHOLD*(2.0/3.0) or angle>360-ANGLE_THRESHOLD*(2.0/3.0) or (135+ANGLE_THRESHOLD*(2.0/3.0)>angle and angle >135-ANGLE_THRESHOLD*(2.0/3.0)) or (225+ANGLE_THRESHOLD*(2.0/3.0)>angle and angle >225-ANGLE_THRESHOLD*(2.0/3.0)):
            return ''    
        if angle<135:
            answer = 'right'
        elif angle<225:
            answer = 'back'
        else:
            answer = 'left'
    elif len(mode.split('&'))==2:
        angle = get_angle(v1,v2)
        if angle<ANGLE_THRESHOLD or angle>360-ANGLE_THRESHOLD or (180+ANGLE_THRESHOLD>angle and angle >180-ANGLE_THRESHOLD):
            return ''
        if angle<180:
            answer = 'right'
        else:
            answer = 'left'
    else:
        angle = get_angle(v1,v2)
        if angle>ANGLE_THRESHOLD//2 and angle<90-ANGLE_THRESHOLD//2:
            answer = 'front-right'
        elif angle>90+ANGLE_THRESHOLD//2 and angle<180-ANGLE_THRESHOLD//2:
            answer = 'rear-right'
        elif angle>180+ANGLE_THRESHOLD//2 and angle<270-ANGLE_THRESHOLD//2:
            answer = 'rear-left'
        elif angle>270+ANGLE_THRESHOLD//2 and angle<360-ANGLE_THRESHOLD//2:
            answer = 'front-left'
        else:
            answer = ''
                
    return answer

def item_rounding(item_,num_digits=2):
    """Rounds numeric values or elements in an array-like object.

    Args:
        item_ (float, int, or array-like): Input number or iterable.
        num_digits (int, optional): Number of decimal places (default: 2).

    Returns:
        Same as input: Rounded number or list/array with rounded elements.

    Example:
        >>> item_rounding(3.14159)
        3.14
        >>> item_rounding([1.234, 5.678])
        [1.23, 5.68]
    """
    if not isinstance(item_,float) and not isinstance(item_,int):
        for i in range(len(item_)):
            item_[i] = round(item_[i],num_digits)
    else:
        item_ = round(item_,num_digits)
    return item_


def distance_between_pcd(pcd1,pcd2):
    """Computes the minimum Euclidean distance between two point clouds or points.

    Args:
        pcd1 (numpy.ndarray): First point or point cloud (shape: (n, 2/3) or (2/3,)).
        pcd2 (numpy.ndarray): Second point or point cloud (shape: (m, 2/3) or (2/3,)).

    Returns:
        float: Minimum Euclidean distance between any pair of points.

    Note:
        - Handles single points (1D) by reshaping to (1, 2/3).
        - Uses scipy.spatial.distance.cdist for efficient computation.

    Example:
        >>> distance_between_pcd([0, 0], [1, 1])
        1.4142135623730951
        >>> distance_between_pcd([[0, 0], [1, 1]], [[2, 2], [3, 3]])
        1.4142135623730951
    """
    if len(pcd1.shape) ==1:
        pcd1 = pcd1.reshape(1,pcd1.shape[0])
    if len(pcd2.shape) ==1:
        pcd2 = pcd2.reshape(1,pcd2.shape[0])
      
    distances = cdist(pcd1, pcd2, 'euclidean')

    min_dist = np.min(distances)
    return min_dist