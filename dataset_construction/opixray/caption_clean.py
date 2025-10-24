import json
import re

def check_bbox_alignment(data):

    if 'bbox' not in data or 'caption' not in data:
        return False

    bboxes_from_field = data['bbox']
    caption = data['caption']


    if not bboxes_from_field:
        return "<bbox>" not in caption
    try:
        dataset_bboxes = {tuple(map(int, bbox)) for bbox in bboxes_from_field}
    except (ValueError, TypeError):
        return False
    bbox_tags = re.findall(r'<bbox>(.*?)</bbox>', caption, re.DOTALL)
    caption_bboxes = set()
    for bbox_str in bbox_tags:
        cleaned_str = bbox_str.replace(" ", "")
        if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
            cleaned_str = cleaned_str[1:-1]
        coords = cleaned_str.split(',')
        if len(coords) != 4:
            return False
            
        try:
            caption_bboxes.add(tuple(map(int, coords)))
        except ValueError:
            return False
    return dataset_bboxes == caption_bboxes


def filter_jsonl(input_path, output_path):

    valid_data = []
    invalid_count = 0
    

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"error -> {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if check_bbox_alignment(data):
                    valid_data.append(data)
                else:
                    invalid_count += 1
            except json.JSONDecodeError:
                print(f"warning!: {line.strip()}")
                invalid_count += 1
            except Exception as e:
                print(f"error: {line.strip()}. is: {e}")
                invalid_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"data_all: {total_lines}")
    print(f"valid: {len(valid_data)}")
    print(f"no valid: {invalid_count}")

input_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption_v2.jsonl'
output_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption_坐标对齐.jsonl'

filter_jsonl(input_file, output_file)
