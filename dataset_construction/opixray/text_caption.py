'''
Built on Qwen3-VL and STING-BEE
'''


import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import math
from typing import Tuple

# --- Start of Added Code for Bbox Resizing ---

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """
    Converts a bounding box from original image coordinates to the coordinates
    of the image after it's been resized by the `smart_resize` logic.
    """
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    # Clamp the coordinates to be within the new image dimensions
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

# --- End of Added Code for Bbox Resizing ---


def setup_distributed():
    """Initializes distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0

class CaptionDataset(Dataset):
    def __init__(self, jsonl_path, processor_instance):
        self.data = []
        self.image_info = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
                image_path = item['image_id']
                if image_path not in self.image_info:
                    try:
                        with Image.open(image_path) as img:
                            self.image_info[image_path] = img.size
                    except FileNotFoundError:
                        print(f"Warning: Image not found at {image_path}. Skipping.")
                        self.image_info[image_path] = None

        self.processor = processor_instance
        # --- MODIFIED PROMPT TEMPLATE ---
        self.prompt_template = """You are an expert in X-ray security analysis. Your task is to generate a detailed, accurate, and fluent descriptive caption for the provided X-ray image based on the given labels and bounding boxes.

Key Principles:
* X-ray Context: Remember that X-ray imaging reveals materials based on density and thickness, leading to unique color representations that differ from natural photographs.
* Dynamic Content: The caption must accurately reflect all provided items. If there is one item, focus on it. If there are multiple, describe them in relation to each other and the overall package.
* Structural Requirements: Your caption must seamlessly integrate the following elements:
  1. Inventory & Count: Clearly state the number and class of all prohibited items present (e.g., "a single lighter," "two items: a sprayer and a power bank").
  2. Precise Location: For each item, embed its exact coordinates in the format <bbox>[x1,y1,x2,y2]<\bbox>.
  3. Spatial Description: Describe the approximate location of each item within the luggage or package (e.g., "in the upper-left corner," "nestled among clothing," "at the center").
  4. Detailed Features: Describe fine-grained characteristics of each item (e.g., "a metal lighter with a transparent fuel chamber," "a sprayer with a red plastic nozzle").
  5. Color & Material: Comment on the X-ray color representation of the items (e.g., "appears orange due to its organic material," "shows a bright blue, indicating dense plastic or metal").
  6. Container Context: Briefly describe the surrounding contents or the container itself (e.g., "inside a toiletry bag," "packed between books," "visible through a grey backpack").

Based on the image, generate a caption for the following prohibited items: [class].
Their corresponding bounding box coordinates are: [bbox].

Output Format:
Generate a single, cohesive paragraph. Please note that the caption must have the contraband <bbox>[x1,y1,x2,y2]<\bbox> field. Vary your sentence structure and vocabulary to ensure each caption is unique and natural-sounding. Output only the caption itself, with no additional explanations.
"""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_id']
        
        if self.image_info.get(image_path) is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        orig_width, orig_height = self.image_info[image_path]
        ground_truth = item['ground_truth']
        bbox_raw = item['bbox']

        # --- START OF MODIFICATION FOR MULTIPLE BBOXES ---
        # Ensure bbox is a list of lists, e.g., [[x1,y1,x2,y2], [x3,y3,x4,y4]]
        if isinstance(bbox_raw, list) and all(isinstance(b, (list, tuple)) for b in bbox_raw):
            bboxes_original = [list(b) for b in bbox_raw]
        elif isinstance(bbox_raw, str):
            import ast
            parsed_bbox = ast.literal_eval(bbox_raw)
            if isinstance(parsed_bbox, list) and all(isinstance(b, (list, tuple)) for b in parsed_bbox):
                 bboxes_original = [list(b) for b in parsed_bbox]
            else: # Handle case like "[[x1,y1,x2,y2]]"
                 bboxes_original = [list(parsed_bbox[0])]
        else:
            raise TypeError(f"bbox must be a list of lists, got {type(bbox_raw)}")

        resized_bboxes = []
        for bbox_original in bboxes_original:
            if len(bbox_original) != 4:
                raise ValueError(f"Each bbox must have 4 elements, got {len(bbox_original)}: {bbox_original}")
            resized_bboxes.append(convert_to_qwen25vl_format(bbox_original, orig_height, orig_width))
        
        class_str = ', '.join(ground_truth)
        bbox_str_list = [f"<bbox>[{','.join(map(str, b))}]</bbox>" for b in resized_bboxes]
        bbox_str = ', '.join(bbox_str_list)
        # --- END OF MODIFICATION ---

        text_prompt = self.prompt_template.replace("[classes]", class_str).replace("[bboxes]", bbox_str)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        return item, messages

def collate_fn(batch, processor_instance):
    items, messages_list = zip(*batch)
    
    texts = [processor_instance.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    
    image_inputs, video_inputs = process_vision_info(messages_list)
    
    inputs = processor_instance(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return items, inputs

def generate_captions_distributed(rank, local_rank, model_path, input_jsonl_path, output_jsonl_path):
   
    is_main_process = (rank == 0)
    processor = AutoProcessor.from_pretrained(model_path)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left" 

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"}
    )

    dataset = CaptionDataset(input_jsonl_path, processor)
    
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
    else:
        print("Running in non-distributed mode. Using a standard DataLoader.")
        sampler = None

    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=(sampler is None), collate_fn=lambda batch: collate_fn(batch, processor), num_workers=4)

    all_results = []
    if is_main_process:
        if dist.is_initialized():
            print(f"Starting inference on {dist.get_world_size()} GPUs.")
        else:
            print("Starting inference on a single GPU.")

    for items, inputs in dataloader:
        inputs = inputs.to(f"cuda:{local_rank}")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for item, caption in zip(items, output_texts):
            result_entry = {
                "image_id": item['image_id'],
                "ground_truth": item['ground_truth'],
                "bbox": item['bbox'],
                "caption": caption
            }
            all_results.append(result_entry)
        
        if is_main_process:
            print(f"Rank {rank} processed a batch.")

    if dist.is_initialized():
        all_results_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_results_gathered, all_results)
        if is_main_process:
            final_results = [item for sublist in all_results_gathered for item in sublist]
    else:
        final_results = all_results
    
    if is_main_process:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for result in final_results:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Processing complete. Results saved to {output_jsonl_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":

    local_model_path = "/home/data2/zkj/llt_code/public_model/Qwen2.5-VL-7B-Instruct/"
    input_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_all.jsonl'
    output_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption_v2.jsonl'

    rank, local_rank = setup_distributed()
    
    generate_captions_distributed(
        rank=rank, 
        local_rank=local_rank,
        model_path=local_model_path,
        input_jsonl_path=input_jsonl_file,
        output_jsonl_path=output_jsonl_file
    )
