import os
import json
import random


IMAGE_DIR = "/home/xray/xray_common/OPIXray/test/test_image"
ANNOTATION_DIR = "/home/xray/xray_common/OPIXray/test/test_annotation"
OUTPUT_JSON = "/home/data2/zkj/llt_code/STING-BEE/stingbee/othermodel_eval/classfication_benchmark/opixray_cls_questions.jsonl"


QUESTION_TEMPLATES = [
    "Which prohibited item category is present in this baggage scan? Select the correct option:\n{options}Note: Only one category is present. Provide a single letter as the answer, e.g., \"A\".",
    "What type of threat is detected in the X-ray image? Choose the correct category:\n{options}There is exactly one correct answer. Respond with a single letter, e.g., \"C\".",
    "Identify the prohibited item category visible in the baggage scan:\n{options}Only one category is present. Output the corresponding letter, e.g., \"B\".",
    "Which of the following categories does the detected threat belong to?\n{options}Select the single correct option by only providing its letter, e.g., \"D\"."
]

def load_all_annotations_and_classes():
    all_classes = set()
    image_class_map = {}

    for ann_file in os.listdir(ANNOTATION_DIR):
        if not ann_file.endswith(".txt"):
            continue
        image_id = ann_file.replace(".txt", "")
        ann_path = os.path.join(ANNOTATION_DIR, ann_file)

        classes_in_image = set()
        with open(ann_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_name = parts[1]
                    classes_in_image.add(cls_name)
                    all_classes.add(cls_name)

        if not classes_in_image:
            print(f"Warning: No valid annotations in {ann_file}")
            continue

        if len(classes_in_image) > 1:
            print(f"Error: Image {image_id} has multiple classes: {classes_in_image}. Skipping.")
            continue

        unique_class = next(iter(classes_in_image))
        image_class_map[image_id] = unique_class

    sorted_classes = sorted(all_classes)
    return sorted_classes, image_class_map

def class_name_to_letter(cls_name, global_class_list):
    if cls_name in global_class_list:
        idx = global_class_list.index(cls_name)
        return chr(ord('A') + idx)
    else:
        raise ValueError(f"Class {cls_name} not in global list")

def build_options_text(global_class_list):
    options = ""
    for i, cls in enumerate(global_class_list):
        letter = chr(ord('A') + i)
        options += f"{letter}. {cls}\n"
    return options

def main():
    print("Loading annotations and collecting classes...")
    global_classes, image_class_map = load_all_annotations_and_classes()
    print(f"Total unique classes: {len(global_classes)}")
    print("Classes:", global_classes)

    options_text = build_options_text(global_classes)
    
    results = []
    print("Generating VQA entries (single-category only)...")

    for image_file in os.listdir(IMAGE_DIR):
        if not image_file.endswith(".jpg"):
            continue
        image_id = image_file.replace(".jpg", "")
        if image_id not in image_class_map:
            continue

        image_path = os.path.abspath(os.path.join(IMAGE_DIR, image_file))
        class_name = image_class_map[image_id]
        gt_letter = class_name_to_letter(class_name, global_classes)
        template = random.choice(QUESTION_TEMPLATES)
        question_text = template.format(options=options_text)

        entry = {
            "image": image_path,
            "text": question_text,
            "category": "Instances Identity",
            "ground_truth": gt_letter
        }
        results.append(entry)


    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Output saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()