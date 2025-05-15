import os
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import argparse

num_mapping = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
skip_type = ['None']


def Judgement_evalution(pred,gt,options):
    """Evaluation for Judgement Questions. 
    Args:
        pred (str): Output of the model.
        gt (str): _description_
        options (list of str): All possible options.
    """
    gt = gt.replace('\"','')
    assert gt in options
    def longest_common_subsequence(str1, str2):
        m, n = len(str1), len(str2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    pred = str(pred)
    option_idx = np.argmax([longest_common_subsequence(pred.strip().lower(), option.strip().lower()) for option in options])
    return float(gt==options[option_idx])

def Estimation_evaluation(pred,gt):
    """Evaluation for Estimation Questions (follow VSI)
    Args:
        pred (_type_): _description_
        gt (_type_): _description_
        anchor (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    try:
        pred = float(pred)
        gt = float(gt)
    except:
        return 0.0
    delta_ratio = abs(gt-pred)/abs(gt)
    citerion_list = [0.5+0.05*i for i in range(10)]
    metric = 0.1*sum([int(delta_ratio<1-citerion) for citerion in citerion_list])
    return metric
 

def Enumeration_evalution(pred,gt):
    """Evaluation for Temporal-Loc and Counting.
    Args:
        pred (_type_): _description_
        gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        if isinstance(pred,str):
            for word in num_mapping:
                if word in pred.lower():
                    pred = num_mapping[word]
                    break
        pred = int(pred)
    except:
        return 0.0
    return float(pred==gt)


class OST_evaluator:
    def __init__(self):
        pass
    def evaluation(self,sample):
 
        if sample['type'] in skip_type or 'pred' not in sample.keys():
            return sample
        
        if 'Estimation' in sample['type']:
            eval_function = Estimation_evaluation
        elif 'Judgement' in sample['type']:
            eval_function = Judgement_evalution
        elif 'Counting' in sample['type'] or 'Temporal-Loc' in sample['type']:
            eval_function = Enumeration_evalution
        else:
            return sample
        
        if 'Judgement' in sample['type']:
            sample['metric'] = eval_function(sample['pred'],sample['answer'],sample['option'])
        else:
            sample['metric'] = eval_function(sample['pred'],sample['answer'])
            
            
        return sample

def process_answer(raw_answer):
    if isinstance(raw_answer,list):
        raw_answer = raw_answer[0]
    try:
        if 'answer' not in raw_answer:
            answer_text = raw_answer
        else:    
            answer_text = raw_answer.split('\"answer\":')[1].split('\"reason\"')[0].split('\n')[0].strip().strip(',').strip()
    except:
        answer_text = raw_answer.split('answer:')[1].split('\n')[0].strip(',').strip()
    answer_text = answer_text.strip('\"')
    return answer_text


# def name_mapping(raw_name):
#     if raw_name == 'A_room-size(float)':
#         return 'None'
#     new_name = raw_name.replace('float','Estimation')

#     if 'quantity' not in new_name:
#         new_name = new_name.replace('int','Temporal-loc')
#     else:
#         new_name = new_name.replace('int','Counting')

#     new_name = new_name.replace('option','Judgement')
#     new_name = new_name.replace('A_object-','Agent_visible_info-')
#     new_name = new_name.replace('AO_','Agent_object_spatial-')
#     new_name = new_name.replace('A_','Agent_state-')
#     return new_name     

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluator of OST-Bench')
    
    parser.add_argument('--result_dir', type=str, default='../results')
    
    args = parser.parse_args()
    
    test_result_dir = args.result_dir
    st_eval = OST_evaluator()
    total_cnt = {}
    samples = []
    correct_cnt = {}
    
    for json_file in os.listdir(test_result_dir):
        samples.extend(json.load(open(os.path.join(test_result_dir,json_file)))['user_message'])
        
    
    
    for idx,sample in tqdm(enumerate(samples)):
        
        sample['type'] = name_mapping(sample['type'])
        sample['pred'] =  process_answer(sample['response'])
        
        
        sample = st_eval.evaluation(sample)
        type = sample['type']
        if 'metric' not in sample.keys():
            continue
        if type not in total_cnt:
            total_cnt[type] = 0
            correct_cnt[type] = 0
        total_cnt[type]+=1
        correct_cnt[type]+=sample['metric']
    print(sorted({k:correct_cnt[k]/total_cnt[k] for k in total_cnt.keys() }.items()))       
            
   