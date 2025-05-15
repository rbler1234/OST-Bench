from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc
import time
import json
from tqdm import tqdm

class Qwen2VL:
    def __init__(self, model_path = None, max_new_tokens = 10024, min_pixels = 128*28*28, max_pixels = 256*28*28):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
        }
    
    def parse_input(self, query=None, imgs=None,system_prompt=None):
        if imgs is None:
            messages = [{"role": "user", "content": query}]
            return messages
        
        if isinstance(imgs, str):
            imgs = [imgs]
        content = []
        if system_prompt is not None:
            content.append({"type": "text", "text": system_prompt})
        for img in imgs:
            content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        return messages

    def chat(self, query = None, imgs = None, history = None,system_prompt=None):
        if history is None:
            history = []
            
        user_query = self.parse_input(query, imgs,system_prompt)
        history.extend(user_query)

        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        history.append({"role": "assistant", "content": response})

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        return response, history 

def process_data(data,image_root,save_root,model_path):
    """
    This function takes conversation data for a specific scan ID, processes the image paths,
    generates responses for each dialogue turn using the QwenVL2.5, and saves
    the enhanced data (including responses) back to a JSON file.

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
        Root directory where the images are stored (will be prepended to image paths)
    save_root : str
        Directory where the output JSON file will be saved
    model_path : str, optional
        The path of pretrain weight

    Returns:
    --------
    None
        The function saves the results to a file but doesn't return anything directly.    
    """
    qwen_agent = Qwen2VL(model_path)
    
    
    scan_id = data['scan_id']
    save_file_name = f'{save_root}/{scan_id}.json'    
    system_prompt = data['system_prompt']
    user_message_list = data['user_message']
    info_list = []
    history = None
    results = []
    start = time.time()
    for user_message in user_message_list:
        image_paths = user_message["image_paths"]
        user_message['response'],history = qwen_agent.chat(user_message["prompt"],image_paths,history,system_prompt)
        
    end = time.time()    
    data['total_time'] = end-start 
    
    with open(save_file_name, 'w') as f: 
        json.dump(data, f, indent=4)
            
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='QwenVL running')
    
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--anno_json_path', type=str, default='../data/OST_bench_v0.json')
    parser.add_argument('--image_root', type=str, default='../data/img/image_upload')
    parser.add_argument('--save_root', type=str, default='../results')
    args = parser.parse_args()
    
    model_path = args.model_path
    anno_json_path = args.anno_json_path
    image_root = args.image_root
    save_root = args.save_root
    os.makdirs(save_root,exist_ok=True)
    all_datas = json.load(open(anno_json_path))
    for data in tqdm(all_datas):
        process_data(data,image_root,save_root,model_path)
    

   
    
   
    