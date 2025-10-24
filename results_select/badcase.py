"""
Used to filter stingbee's badcase, 
other models did not show data for this badcase.
"""


import json

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item['image_id']
            answer = item['answer']
            gt = item['ground_truth']
            data[image_id] = {
                'answer': answer,
                'ground_truth': gt
            }
    return data

def normalize_response(resp):
    """Normalize response string for comparison: sort letters and format consistently."""
    if not resp:
        return ""
    letters = [x.strip().upper() for x in resp.split(',')]
    letters = sorted([x for x in letters if x])
    return ', '.join(letters)

data_A = load_jsonl('A.jsonl')
data_B = load_jsonl('B.jsonl')

badcases = []

for image_id in data_A:
    if image_id not in data_B:
        continue

    item_A = data_A[image_id]
    item_B = data_B[image_id]

    ans_A = normalize_response(item_A['answer'])
    gt_A = normalize_response(item_A['ground_truth'])

    ans_B = normalize_response(item_B['answer'])
    gt_B = normalize_response(item_B['ground_truth'])
    if ans_A == gt_A and ans_B != gt_B:
        badcases.append({
            'image_id': image_id,
            'good_response': item_A['answer'],
            'bad_response': item_B['answer']
        })

with open('badcase.jsonl', 'w', encoding='utf-8') as f:
    for case in badcases:
        f.write(json.dumps(case, ensure_ascii=False) + '\n')

print(f"Found {len(badcases)} bad cases.")