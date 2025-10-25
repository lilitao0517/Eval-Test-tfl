import json
import re
import math


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """
    Smartly resizes an image to have dimensions that are multiples of `factor`,
    while keeping the total number of pixels within a specified range.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """
    Converts a bounding box from original image coordinates to the coordinates
    of the image after it's been resized by the `smart_resize` logic.
    """
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def process_and_validate_data(data):
    """
    Check and process the data:
    1. if there are no <bbox> tags in caption, keep them directly.
    2. if there is, check if the number of bbox fields and <bbox> tags in caption are equal.
    3. if equal, iterate through all bboxes, perform coordinate conversion, and replace the tags in caption in order.
    4. return the processed data, or None if invalid.
    """
    if 'bbox' not in data or 'caption' not in data:
        return None
    caption = data['caption']
    bboxes_from_field = data['bbox']

    if "<bbox>" not in caption:
        return data


    if len(bboxes_from_field) == 0:
        return None

    bbox_tags_in_caption = re.findall(r'<bbox>(.*?)</bbox>', caption, re.DOTALL)
    if len(bboxes_from_field) != len(bbox_tags_in_caption):
        return None

    try:
        original_bboxes = [list(map(int, bbox)) for bbox in bboxes_from_field]
        max_x2 = max(bbox[2] for bbox in original_bboxes)
        max_y2 = max(bbox[3] for bbox in original_bboxes)
        orig_height = max_y2
        orig_width = max_x2

        if orig_height == 0 or orig_width == 0:
            return None


        updated_caption = caption
        for original_bbox in original_bboxes:

            new_bbox = convert_to_qwen25vl_format(original_bbox, orig_height, orig_width)
            new_bbox_str = f"[{', '.join(map(str, new_bbox))}]"
            updated_caption = re.sub(r'<bbox>.*?</bbox>', f'<bbox>{new_bbox_str}</bbox>', updated_caption, count=1, flags=re.DOTALL)
        

        new_data = data.copy()
        new_data['caption'] = updated_caption
        return new_data

    except (ValueError, TypeError, IndexError, KeyError) as e:

        print(f"Warning: Failed to process data due to error: {e}. Data: {data}")
        return None



def filter_jsonl(input_path, output_path):
    valid_data = []
    invalid_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_path}")
        return

    print(f"Processing {total_lines} lines from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                processed_data = process_and_validate_data(data)
                if processed_data:
                    valid_data.append(processed_data)
                else:
                    invalid_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON format on line {i+1}: {line.strip()}")
                invalid_count += 1
            except Exception as e:
                print(f"Error on line {i+1}: {e}. Line content: {line.strip()}")
                invalid_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n--- Processing Complete ---")
    print(f"Total lines in input: {total_lines}")
    print(f"Valid and processed lines: {len(valid_data)}")
    print(f"Invalid or filtered lines: {invalid_count}")
    print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    input_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption.jsonl'
    output_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/opixray/trainset_caption_坐标筛选.jsonl'

    filter_jsonl(input_file, output_file)
