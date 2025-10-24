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
        self.prompt_template = (
            "You are an expert in X-ray security screening, specializing in generating descriptive captions for X-ray security images."
            "X-ray imaging is based on material density and thickness, resulting in color representations that differ significantly from natural images."
            "The image above shows a prohibited item belonging to the class [class], with its bounding box coordinates given as [bbox]."
            "These coordinates are specified using the top-right and bottom-left corners of the bounding box."
            "Please generate a detailed and accurate image caption that describes the following aspects: the prohibited item's class, its approximate location within the package, its fine-grained features, its color representation, its precise location (coordinates—formatted in the caption as <bbox>[x1,y1,x2,y2]<bbox>), and the characteristics of the package containing it."
            "Here is an example caption:"
            "A utility knife with a blue-green tinted blade and handle is visible in the transparent plastic container in the center of the package at <bbox>[526, 389, 658, 457]<bbox>, with clear grid lines on the surface of the container."

            "Captions should be diverse in phrasing and fluent in expression. Please output only the caption itself, with no additional text."
        )

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

    # Create dataset and dataloader
    dataset = CaptionDataset(input_jsonl_path, processor)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    
    # Use a lambda to pass the processor instance to the collate_fn
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

    # Gather results from all processes
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
    input_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_all.jsonl'
    output_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption_v2.jsonl'

    # torchrun --nproc_per_node=4 dataset_construction/opixray/text_caption.py
    generate_captions_distributed(
        rank=0, 
        local_rank=0,
        model_path=local_model_path,
        input_jsonl_path=input_jsonl_file,
        output_jsonl_path=output_jsonl_file
    )
