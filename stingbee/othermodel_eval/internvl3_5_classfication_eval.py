import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import numpy as np
from modelscope import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    print(f"Loading model from {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)


    print(f"Loading questions from {args.question_file}")
    with open(os.path.expanduser(args.question_file), "r") as f:
        lines = f.readlines()
    questions = [json.loads(q) for q in lines]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        pixel_values_list = []
        num_patches_list = []
        prompts = []


        for q in batch_questions:
            image_file = q['image']
            pixel_values = load_image(image_file, input_size=448, max_num=12).to(torch.bfloat16).cuda()
            num_patches = pixel_values.size(0)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(num_patches)


            qs = q['text']
            prompt = "<image>\n" + qs 
            prompts.append(prompt)

        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
        try:
            responses = model.batch_chat(
                tokenizer,
                pixel_values_batch,
                num_patches_list=num_patches_list,
                questions=prompts,
                generation_config=dict(
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
            )
        except Exception as e:
            print(f"Error during batch inference: {e}")
            responses = [""] * len(prompts)

        for idx, q in enumerate(batch_questions):
            output = responses[idx].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "image_id": q["image"],
                "answer": output,
                "ground_truth": q['ground_truth'],
                "question": q['text'],
                "type": "classfication",
                "dataset": "Pidray"
            }) + "\n")
            ans_file.flush()

    ans_file.close()
    print(f"Results saved to {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/data2/zkj/llt_code/public_model/InternVL3_5-1B")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)