import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    print(f"Loading Qwen2.5-VL model from {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", 
        device_map="auto",
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,   
        max_pixels=args.max_pixels,   
        trust_remote_code=True
    )

    print(f"Loading questions from {args.question_file}")
    with open(os.path.expanduser(args.question_file), "r") as f:
        lines = f.readlines()
    questions = [json.loads(q) for q in lines]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        batch_messages = []
        batch_images = []

        for q in batch_questions:
            image_file = q['image']
            image = Image.open(image_file).convert('RGB')
            qs = q['text']
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": qs}
                    ]
                }
            ]
            batch_messages.append(messages)
        try:
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs_list = []
            for msg in batch_messages:
                img_inputs, _ = process_vision_info(msg)
                image_inputs_list.append(img_inputs[0] if img_inputs else None)

            inputs = processor(
                text=texts,
                images=image_inputs_list,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p
            )

            input_ids = inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        except Exception as e:
            print(f"Error during batch inference: {e}")
            outputs = [""] * len(batch_questions)

        for idx, q in enumerate(batch_questions):
            output = outputs[idx].strip()
            ans_file.write(json.dumps({
                "image_id": q["image"],
                "answer": output,
                "ground_truth": q['ground_truth'],
                "question": q['text'],
                "type": "classification",
                "dataset": "Pidray"
            }) + "\n")
            ans_file.flush()

    ans_file.close()
    print(f"Results saved to {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_pixels", type=int, default=448 * 448)          
    parser.add_argument("--max_pixels", type=int, default=12 * 448 * 448) 

    args = parser.parse_args()

    eval_model(args)