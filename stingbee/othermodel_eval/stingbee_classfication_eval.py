import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import sys

sys.path.append("/home/data2/zkj/llt_code/STING-BEE/")
from stingbee.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from stingbee.conversation import conv_templates, SeparatorStyle
from stingbee.model.builder import load_pretrained_model
from stingbee.utils import disable_torch_init
from stingbee.mm_utils import tokenizer_image_token, get_model_name_from_path


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()

    # Load questions
    with open(os.path.expanduser(args.question_file), "r") as f:
        lines = f.readlines()
    questions = [json.loads(q) for q in lines]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        input_ids_list = []
        images_list = []
        stop_strs = []

        for q in batch_questions:
            image_file = q['image']
            qs = q['text']

            # Build prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()



            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
            input_ids = input_ids.unsqueeze(0)  
            input_ids_list.append(input_ids)
            image = Image.open(image_file).convert('RGB')
            images_list.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_strs.append(stop_str)

        # Pad input_ids
        max_len = max(t.size(1) for t in input_ids_list)
        padded_input_ids = []
        for t in input_ids_list:
            pad_len = max_len - t.size(1)
            padded = torch.cat([
                torch.full((1, pad_len), tokenizer.pad_token_id, device=t.device, dtype=t.dtype),
                t
            ], dim=1)
            padded_input_ids.append(padded)
        input_ids_batch = torch.cat(padded_input_ids, dim=0)
        image_tensor_batch = image_processor.preprocess(images_list,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values'].half().cuda()

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_batch,
                images=image_tensor_batch,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=256,
                use_cache=True
            )

        input_token_len = input_ids_batch.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        for idx, q in enumerate(batch_questions):
            output = outputs[idx].strip()
            if output.endswith(stop_strs[idx]):
                output = output[:-len(stop_strs[idx])]
            output = output.strip()

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
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)