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
    """Evaluates model performance on judgement questions using longest common subsequence matching.

    This function compares the model's predicted answer against ground truth by finding the 
    option with the highest textual similarity using longest common subsequence (LCS) algorithm.

    Args:
        pred (str): Model's predicted answer string.
        gt (str): Ground truth answer string (must match one of the provided options).
        options (list[str]): List of all possible valid answer choices.

    Returns:
        float: 1.0 if the best-matching option equals ground truth, 0.0 otherwise.
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
    """Evaluates numerical estimation answers using graduated accuracy scoring (follow VSI-bench).

    Implements a tiered scoring system where closer estimates receive higher scores:
    - Perfect match: 1.0
    - Within 5%: 0.9
    - Within 10%: 0.8
    - ... down to 0.1 for marginal matches

    Args:
        pred (str/float): Model's predicted numerical value.
        gt (str/float): Ground truth numerical value.

    Returns:
        float: Score between 0.0  and 1.0 based on relative error.
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
    """Evaluates enumeration-type answers (counts/temporal locations) with word-to-number conversion.

    Handles both numeric inputs and textual number representations (e.g., "three" → 3)
    using a predefined number mapping dictionary (num_mapping).

    Args:
        pred (str/int): Model's predicted answer (either number or number word).
        gt (str/int): Ground truth numerical value.

    Returns:
        float: 1.0 if prediction matches ground truth after conversion, 0.0 otherwise.
    """
    try:
        if isinstance(pred,str):
            for word in num_mapping:
                if word in pred.lower():
                    pred = num_mapping[word]
                    break
        pred = int(pred)
        gt = int(gt)
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
        elif 'Counting' in sample['type'] or 'Temporal-loc' in sample['type']:
            eval_function = Enumeration_evalution
        else:
            return sample
        
        if 'Judgement' in sample['type']:
            sample['metric'] = eval_function(sample['pred'],sample['answer'],sample['option'])
        else:
            sample['metric'] = eval_function(sample['pred'],sample['answer'])
            
            
        return sample

def process_answer(raw_answer):
    """
    Processes and extracts the 'answer' portion from a raw model response string or list.
    
    This function handles multiple response formats to robustly extract the answer text:
    - Handles both string inputs and single-element lists
    - Supports both JSON-style (key:value) and plain text responses
    - Cleans extracted answers by removing surrounding quotes and whitespace
    
    Args:
        raw_answer (str or list): The raw model response to process. If a list is provided,
                                only the first element will be processed. Expected formats:
                                - JSON-style: '{"answer": "text", "reason": "..."}'
                                - Key-value style: 'answer: text reason: ...'
                                - Plain text (returned as-is)
    
    Returns:
        str: The extracted answer text with surrounding quotes and whitespace removed.
             Returns empty string if answer field cannot be found, and returns the
             raw input if neither 'answer' nor 'reason' fields are detected.
    
    Note:
        - If parsing fails, prints an error message with the problematic input
        - Maintains backward compatibility by returning plain text inputs unchanged
        - Strips trailing commas that may appear in JSON parsing
    
    Example:
        >>> process_answer('{"answer": "yes", "reason": "because..."}')
        'yes'
        >>> process_answer('answer: yes reason: because...')
        'yes'
        >>> process_answer('simple text response')
        'simple text response'
    """
    if isinstance(raw_answer,list):
        raw_answer = raw_answer[0]
   
    raw_answer = raw_answer.replace('\'','\"')
    try:
        if 'reason' not in raw_answer or 'answer' not in raw_answer:
            answer_text = raw_answer
        elif '\"answer\":' in raw_answer:
            answer_text = raw_answer.split('\"answer\":')[1].split('\"reason\"')[0].split('\n')[0].strip().strip(',').strip()
        elif 'answer:' in raw_answer:
            answer_text = raw_answer.split('answer:')[1].split('reason')[0].split('\n')[0].strip().strip(',').strip()
        else:
            answer_text = ''
        answer_text = answer_text.strip('\"')
            
    except:
        print('answer format error:',raw_answer) 
    return answer_text



def collect_results(static_results):
    """
    Aggregates and computes composite metrics from individual evaluation results.

    This function processes raw evaluation metrics by:
    1. Calculating averaged scores for multi-part evaluations
    2. Grouping related metrics into composite categories
    3. Computing overall scores across different evaluation dimensions

    Args:
        static_results (dict): Dictionary containing raw evaluation metrics with keys in format:
            - 'Category_metric-type(EvaluationTypeN)' (individual items)
            - e.g., 'Agent_object_spatial-direction(Judgement1)'

    Returns:
        tuple: (full_dict, overall_dict) where:
            full_dict (dict): Contains all original metrics plus averaged multi-part scores:
                - Maintains all original non-averaged metrics
                - Adds averaged versions for multi-part evaluations:
                    * 'Agent_object_spatial-direction(Judgement)'
                    * 'Agent_object_spatial-distance(Judgement)'
                    * 'Agent_visible_info-existence(Temporal-loc)'

            overall_dict (dict): Contains composite scores grouped by evaluation dimension:
                - 'A_state': Agent state metrics (averaged)
                - 'A_info': Agent visible information metrics (averaged)
                - 'AO': Agent-Object spatial metrics (averaged)
                - Category summaries ('Judgement', 'Estimation', etc.)
                - 'Overall': Grand average of all composite scores
    """
    
    full_dict = {}
    full_dict['Agent_object_spatial-direction(Judgement)'] = sum([static_results[f'Agent_object_spatial-direction(Judgement{i})'] for i in range(1,4)])/3.0
    full_dict['Agent_object_spatial-distance(Judgement)'] = sum([static_results[f'Agent_object_spatial-distance(Judgement{i})'] for i in range(1,4)])/3.0
    full_dict['Agent_visible_info-existence(Temporal-loc)'] = (static_results['Agent_visible_info-existence(Temporal-loc1)'] + static_results['Agent_visible_info-existence(Temporal-loc2)'] )/2.0
    for type_name in static_results:
        if 'Agent_object_spatial-direction(Judgement' in type_name or 'Agent_object_spatial-distance(Judgement' in type_name or 'Agent_visible_info-existence(Temporal-loc' in type_name:
            continue
        full_dict[type_name] = static_results[type_name]
    
    overall_dict = {}
    overall_dict['A_state(Judge)'] = (full_dict['Agent_state-orientation(Judgement)']+full_dict['Agent_state-position(Judgement)'])/2.0
    overall_dict['A_state(Esti)'] = (full_dict['Agent_state-orientation(Estimation)']+full_dict['Agent_state-position(Estimation)'])/2.0
    
    overall_dict['A_info(Judge)'] = (full_dict['Agent_visible_info-existence(Judgement)']+full_dict['Agent_visible_info-order(Judgement)']+\
                                    full_dict['Agent_visible_info-diversity(Judgement)'])/3.0
    overall_dict['A_info(Temp)'] = full_dict['Agent_visible_info-existence(Temporal-loc)']
    overall_dict['A_info(Count)'] = full_dict['Agent_visible_info-quantity(Counting)']
    
    overall_dict['AO(Judge)'] = (full_dict['Agent_object_spatial-direction(Judgement)']+full_dict['Agent_object_spatial-distance(Judgement)'])/2.0
    overall_dict['AO(Esti)'] = (full_dict['Agent_object_spatial-direction(Estimation)']+full_dict['Agent_object_spatial-distance(Estimation)'])/2.0
    overall_dict['AO(Temp)'] = (full_dict['Agent_object_spatial-direction(Temporal-loc)']+full_dict['Agent_object_spatial-distance(Temporal-loc)'])/2.0
    
    overall_dict['Judgement'] = (full_dict['Agent_object_spatial-direction(Judgement)']+full_dict['Agent_object_spatial-distance(Judgement)']+\
                                 full_dict['Agent_visible_info-existence(Judgement)']+full_dict['Agent_visible_info-order(Judgement)']+\
                                full_dict['Agent_visible_info-diversity(Judgement)']+full_dict['Agent_state-orientation(Judgement)']+\
                                full_dict['Agent_state-position(Judgement)'])/7.0
    overall_dict['Estimation'] = (full_dict['Agent_object_spatial-direction(Estimation)']+full_dict['Agent_object_spatial-distance(Estimation)']+\
                                 full_dict['Agent_state-orientation(Estimation)']+\
                                full_dict['Agent_state-position(Estimation)'])/4.0
    overall_dict['Temporal-loc'] = (full_dict['Agent_visible_info-existence(Temporal-loc)']+\
                                    full_dict['Agent_object_spatial-direction(Temporal-loc)']+\
                                    full_dict['Agent_object_spatial-distance(Temporal-loc)'])/3.0
    overall_dict['Counting'] = full_dict['Agent_visible_info-quantity(Counting)']
    
    overall_dict['Overall'] = (overall_dict['A_state(Judge)']+overall_dict['A_state(Esti)']+\
                                overall_dict['A_info(Judge)']+overall_dict['A_info(Temp)']+\
                                overall_dict['A_info(Count)']+overall_dict['AO(Judge)']+\
                                overall_dict['AO(Esti)']+overall_dict['AO(Temp)'])/8.0
    
    return full_dict,overall_dict
    
    
    
        

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Evaluator of OST-Bench')
    
    parser.add_argument('--output_dir', type=str, default='../results')
    
    args = parser.parse_args()
    
    test_result_dir = args.output_dir
    st_eval = OST_evaluator()
    total_cnt = {}
    samples = []
    correct_cnt = {}
    
    for json_file in os.listdir(test_result_dir):
        samples.extend(json.load(open(os.path.join(test_result_dir,json_file)))['user_message'])
    sum_ = 0
    for idx,sample in tqdm(enumerate(samples)):

        sample['response'] = str(sample['response'])
        
        if len(sample['response'])>0:
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
        sum_+=1
    static_results = {k:correct_cnt[k]/total_cnt[k] for k in total_cnt.keys() }
    full_dict,overall_dict = collect_results(static_results)
    
    print('-------------------------- Evaluation Result--------------------------')
    print('Total Samples:',sum_)
    print('Overall Accuracy:',overall_dict['Overall'])
    print('-----------------------------------------------------------------------')
    print('Judgement Accuracy:',overall_dict['Judgement'])
    print('Estimation Accuracy:',overall_dict['Estimation'])
    print('Temporal-loc Accuracy:',overall_dict['Temporal-loc'])
    print('Counting Accuracy:',overall_dict['Counting'])
    print('-----------------------------------------------------------------------')
    print('Agent State(Judgement) Accuracy:', overall_dict['A_state(Judge)'])
    print('Agent State(Estimation) Accuracy:', overall_dict['A_state(Esti)'])
    print('Agent Visible Info(Judgement) Accuracy:', overall_dict['A_info(Judge)'])
    print('Agent Visible Info(Temporal-loc) Accuracy:', overall_dict['A_info(Temp)'])
    print('Agent Visible Info(Counting) Accuracy:', overall_dict['A_info(Count)'])
    print('Agent Object Spatial(Judgement) Accuracy:', overall_dict['AO(Judge)'])
    print('Agent Object Spatial(Estimation) Accuracy:', overall_dict['AO(Esti)'])
    print('Agent Object Spatial(Temporal-loc) Accuracy:', overall_dict['AO(Temp)'])
    print('-----------------------------------------------------------------------')

       
   