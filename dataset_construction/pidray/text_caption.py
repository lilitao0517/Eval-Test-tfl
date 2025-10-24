import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
       
        return 0, 0

class CaptionDataset(Dataset):
    def __init__(self, jsonl_path, processor_instance):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor_instance
        
        self.prompt_template = """You are an expert in X-ray security analysis. Your task is to generate a detailed, accurate, and fluent descriptive caption for the provided X-ray image based on the given labels and bounding boxes.

Key Principles:
* X-ray Context: Remember that X-ray imaging reveals materials based on density and thickness, leading to unique color representations that differ from natural photographs.
* Dynamic Content: The caption must accurately reflect all provided items. If there is one item, focus on it. If there are multiple, describe them in relation to each other and the overall package.
* Structural Requirements: Your caption must seamlessly integrate the following elements:
  1. Inventory & Count: Clearly state the number and class of all prohibited items present (e.g., "a single lighter," "two items: a sprayer and a power bank").
  2. Precise Location: For each item, embed its exact coordinates in the format <bbox>[x1,y1,x2,y2]<bbox>.
  3. Spatial Description: Describe the approximate location of each item within the luggage or package (e.g., "in the upper-left corner," "nestled among clothing," "at the center").
  4. Detailed Features: Describe fine-grained characteristics of each item (e.g., "a metal lighter with a transparent fuel chamber," "a sprayer with a red plastic nozzle").
  5. Color & Material: Comment on the X-ray color representation of the items (e.g., "appears orange due to its organic material," "shows a bright blue, indicating dense plastic or metal").
  6. Container Context: Briefly describe the surrounding contents or the container itself (e.g., "inside a toiletry bag," "packed between books," "visible through a grey backpack").

Based on the image, generate a caption for the following prohibited items: [class].
Their corresponding bounding box coordinates are: [bbox].

Output Format:
Generate a single, cohesive paragraph. Do not use lists or bullet points. Vary your sentence structure and vocabulary to ensure each caption is unique and natural-sounding. Output only the caption itself, with no additional explanations.
"""



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_id']
        ground_truth = item['ground_truth']
        bbox = item['bbox']

        class_str = ', '.join(ground_truth)
        bbox_str = ', '.join(map(str, bbox))
        text_prompt = self.prompt_template.replace("[class]", class_str).replace("[bbox]", bbox_str)

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
    rank, local_rank = setup_distributed()
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
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=lambda batch: collate_fn(batch, processor), num_workers=4)

    all_results = []
    if is_main_process:
        print(f"Starting inference on {dist.get_world_size()} GPUs.")

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


    all_results_gathered = [None] * dist.get_world_size()
    dist.all_gather_object(all_results_gathered, all_results)
    
    if is_main_process:
        final_results = [item for sublist in all_results_gathered for item in sublist]
        
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for result in final_results:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Processing complete. Results saved to {output_jsonl_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    local_model_path = "/home/data2/zkj/llt_code/public_model/Qwen2.5-VL-7B-Instruct/"
    input_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/pidray/trainset_all.jsonl'
    output_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/pidxray/trainset_caption.jsonl'

    generate_captions_distributed(
        rank=0, 
        local_rank=0,
        model_path=local_model_path,
        input_jsonl_path=input_jsonl_file,
        output_jsonl_path=output_jsonl_file
    )

# Start command: torchrun --nproc_per_node=6 dataset_construction/opixray/text_caption.py