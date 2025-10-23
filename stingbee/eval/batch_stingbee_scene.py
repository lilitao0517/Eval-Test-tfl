import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import sys
sys.path.append("/home/data2/zkj/llt_code/STING-BEE/")  # 注意：是 stingbee 的父目录！

from stingbee.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from stingbee.conversation import conv_templates, SeparatorStyle
from stingbee.model.builder import load_pretrained_model
from stingbee.utils import disable_torch_init
from stingbee.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# Define your set of possible classes
classes = ['gun', 'knife', 'pliers', 'wrench', 'scissors', 'nonthreat'] 

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def convert_to_binary_vector(label):
    # Ensure it's a list, split by comma if there are multiple labels
    labels = label.split(',')
          
    # Initialize a binary vector for all classes
    binary_vector = [0] * len(classes)
    valid_labels_found = False  # Track whether any valid labels were found
    
    # Set 1 for present labels, convert to lowercase and check if label exists in the class list
    for lbl in labels:
        lbl = lbl.lower().strip()  # Convert to lowercase and strip any extra spaces     

        if lbl in classes:
            idx = classes.index(lbl)
            binary_vector[idx] = 1
            valid_labels_found = True  # A valid label was found
        else:
            print(f"[Warning] Label '{lbl}' is not a valid class.")

    # If no valid labels are found, return None to signal that the prediction should be discarded
    if not valid_labels_found:
        return None

    return binary_vector

def evaluation_metrics(data_path):
    base = [json.loads(q) for q in open(data_path, "r")]
    
    y_true = []
    y_pred = []
    
    for answers in tqdm(base):
        # Normalize the ground truth and predicted answer (lowercase and strip extra spaces)
        ground_truth = ' '.join(answers['ground_truth'].lower().strip().split())
        answer = ' '.join(answers['answer'].lower().strip().split())

               
        # Convert both to binary vectors
        ground_truth_vector = convert_to_binary_vector(ground_truth)
        answer_vector = convert_to_binary_vector(answer)

        # Discard invalid predictions (when answer_vector is None)
        if answer_vector is None:
            print(f"Invalid prediction found for entry: {answers['question_id']} - Ignored.")
            continue
        
        # Append to the lists for evaluation
        y_true.append(ground_truth_vector)
        y_pred.append(answer_vector)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Calculate mAP (Mean Average Precision)
    mAP = average_precision_score(y_true, y_pred, average='macro')  

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('Mean Average Precision (mAP):', mAP)
    

            

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    # print(model)
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for i in tqdm(range(0,len(questions),args.batch_size)):
        input_batch=[]
        input_image_batch=[]
        count=i
        image_folder=[]     
        batch_end = min(i + args.batch_size, len(questions))

             
        for j in range(i,batch_end):
            image_file=questions[j]['image']
            qs=questions[j]['text']
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)

            image = Image.open(os.path.join(args.image_folder, image_file))

            image_folder.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
        final_input_tensors=torch.cat(final_input_list,dim=0)
        image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values']

        with torch.inference_mode():
            output_ids = model.generate( final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)

        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for k in range(0,len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()
            
            ans_file.write(json.dumps({

                                    "question_id": questions[count]["question_id"],
                                    "image_id": questions[count]["image"],
                                    "answer": output,
                                    "ground_truth": questions[count]['ground_truth']
                                    }) + "\n")
            count=count+1
            ans_file.flush()
    ans_file.close()
    evaluation_metrics(answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
