import argparse
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from modelscope import pipeline


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    print(f"Loading LLaVA-1.5 via ModelScope pipeline from {args.model_path}")
    pipe = pipeline("image-text-to-text", model=args.model_path, device="cuda")

    with open(os.path.expanduser(args.question_file), "r") as f:
        lines = f.readlines()
    questions = [json.loads(q) for q in lines]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for q in tqdm(questions):

  
        image = Image.open(q['image']).convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},      
                    {"type": "text", "text": q['text']}  
                ]
            }
        ]

        outputs = pipe(
            text=messages,
            generate_kwargs={
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        )

        generated = outputs[0]['generated_text']
        output = generated[1]['content'].strip()

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
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    eval_model(args)