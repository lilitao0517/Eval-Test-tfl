import os
import json
def convert_txt_to_jsonl(base_path, output_file):
    image_dir = os.path.join(base_path, 'train_image')
    annotation_dir = os.path.join(base_path, 'train_annotation')
    if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir):
        print(f"Error: Directory not found '{image_dir}' or '{annotation_dir}'")
        return
    all_annotations = {}
    for txt_filename in os.listdir(annotation_dir):
        if not txt_filename.endswith('.txt'):
            continue
        txt_path = os.path.join(annotation_dir, txt_filename)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                image_name = parts[0]
                ground_truth = parts[1]
                try:
                    bbox = list(map(int, parts[2:6]))
                except ValueError:
                    continue
                image_id = os.path.join(image_dir, image_name)
                if image_id not in all_annotations:
                    all_annotations[image_id] = {
                        "image_id": image_id,
                        "ground_truth": [],
                        "bbox": []
                    }
                all_annotations[image_id]["ground_truth"].append(ground_truth)
                all_annotations[image_id]["bbox"].append(bbox)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for data in all_annotations.values():
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
dataset_base_path = '/home/xray/xray_common/OPIXray/train'
output_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_all.jsonl'
convert_txt_to_jsonl(dataset_base_path, output_jsonl_file)
