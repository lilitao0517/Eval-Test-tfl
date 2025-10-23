import os
import json
import random
IMAGE_ROOT = "/home/xray/xray_common/pidray/pidray"
ANNOTATION_ROOT = "/home/xray/xray_common/pidray/pidray/annotations"
OUTPUT_JSON = "/home/data2/zkj/llt_code/STING-BEE/stingbee/othermodel_eval/classfication_benchmark/pidray_multilabel_questions.jsonl"

ANNOTATION_FILES = {
    "easy": "xray_test_easy.json",
    "hidden": "xray_test_hidden.json",
    "hard": "xray_test_hard.json"
}

QUESTION_TEMPLATES = [
    "Which prohibited item categories are present in this baggage scan? Select all that apply:\n{options}\nNote: Multiple categories may be present. Respond with ONLY the letters of the correct options, separated by commas and a space (e.g., \"A, B\"). Do not include any other text.",
    
    "What types of threats are detected in the X-ray image? Choose all correct categories:\n{options}\nMultiple selections are allowed. Output ONLY the corresponding letters in alphabetical order, separated by \", \" (e.g., \"C, D\"). No additional words or punctuation.",
    
    "Identify all prohibited item categories visible in the baggage scan:\n{options}\nList all applicable letters in alphabetical order, separated by \", \" (e.g., \"A, C\"). Your response must contain ONLY these letters and commas—nothing else.",
    
    "Which of the following categories appear in the image? Select all that apply:\n{options}\nProvide ONLY the letters of the detected categories, sorted alphabetically and separated by \", \" (e.g., \"B, D\"). Any extra characters will be considered incorrect."
]

def load_all_pidray_data():

    all_classes = set()
    image_categories = {}

    for split, ann_file in ANNOTATION_FILES.items():
        ann_path = os.path.join(ANNOTATION_ROOT, ann_file)
        if not os.path.isfile(ann_path):
            print(f"Warning: Annotation file not found: {ann_path}")
            continue

        with open(ann_path, 'r') as f:
            coco_data = json.load(f)

        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        all_classes.update(cat_id_to_name.values())

        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data.get('images', [])}

        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id not in image_id_to_filename or category_id not in cat_id_to_name:
                continue

            file_name = image_id_to_filename[image_id]
            category_name = cat_id_to_name[category_id]
            if file_name not in image_categories:
                image_categories[file_name] = set()
            image_categories[file_name].add(category_name)

    sorted_classes = sorted(all_classes)
    return sorted_classes, image_categories

def categories_to_letters(category_set, global_class_list):
    indices = []
    for cls in category_set:
        if cls in global_class_list:
            indices.append(global_class_list.index(cls))
    indices.sort() 
    letters = [chr(ord('A') + i) for i in indices]
    return ", ".join(letters)

def build_options_text(global_class_list):
    options = ""
    for i, cls in enumerate(global_class_list):
        letter = chr(ord('A') + i)
        options += f"{letter}. {cls}\n"
    return options

def main():
    print("Loading Pidray COCO annotations (multi-label mode)...")
    global_classes, image_categories = load_all_pidray_data()
    print(f"Total unique classes: {len(global_classes)}")
    print("Classes:", global_classes)
    options_text = build_options_text(global_classes)
    results = []
    print("Generating multi-label VQA entries...")
    for file_name, category_set in image_categories.items():
        if file_name.startswith("xray_easy"):
            split = "easy"
        elif file_name.startswith("xray_hidden"):
            split = "hidden"
        elif file_name.startswith("xray_hard"):
            split = "hard"
        else:
            print(f"Warning: Cannot determine split for {file_name}")
            continue

        image_path = os.path.abspath(os.path.join(IMAGE_ROOT, split, file_name))
        if not os.path.isfile(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        gt_letters = categories_to_letters(category_set, global_classes)
        if not gt_letters:
            print(f"Warning: No valid categories for {file_name}")
            continue
        template = random.choice(QUESTION_TEMPLATES)
        question_text = template.format(options=options_text)

        entry = {
            "image": image_path,
            "text": question_text,
            "category": "Instances Identity",
            "ground_truth": gt_letters  
        }
        results.append(entry)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Output saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()