import os
import json

def convert_coco_to_jsonl(base_path, output_file):
    """
    将COCO格式的数据集转换为JSONL格式。

    Args:
        base_path (str): 数据集的根目录，应包含图片文件夹和标注JSON文件。
        output_file (str): 输出的JSONL文件路径。
    """
    # 1. 定义文件和文件夹路径
    image_dir = os.path.join(base_path, 'train')  # 存放图片的文件夹
    annotation_file = os.path.join(base_path, 'annotations', 'xray_train.json')  # COCO格式的标注文件

    # 2. 检查路径是否存在
    if not os.path.isdir(image_dir):
        print(f"错误：图片目录未找到 -> '{image_dir}'")
        return
    if not os.path.isfile(annotation_file):
        print(f"错误：标注文件未找到 -> '{annotation_file}'")
        return

    # 3. 加载COCO格式的标注文件
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # COCO格式的核心部分
    images_info = coco_data.get('images', [])
    annotations_info = coco_data.get('annotations', [])
    categories_info = coco_data.get('categories', [])

    # 创建一个从category_id到category_name的映射，方便查找
    id_to_name = {cat['id']: cat['name'] for cat in categories_info}

    # 4. 按图片ID聚合所有标注
    # 先创建一个以图片ID为键的字典，方便快速查找
    annotations_by_image_id = {}
    for ann in annotations_info:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    # 5. 构建最终的数据并写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for img_info in images_info:
            image_id = img_info['id']
            image_filename = img_info['file_name']
            
            # 构建完整的图片路径，作为新的唯一标识符
            image_path = os.path.join(image_dir, image_filename)

            # 获取当前图片的所有标注，如果没有则为空列表
            current_annotations = annotations_by_image_id.get(image_id, [])

            # 提取ground_truth（类别名）和bbox（边界框）
            ground_truths = []
            bboxes = []
            for ann in current_annotations:
                # 从映射中查找类别名
                category_name = id_to_name.get(ann['category_id'], 'unknown')
                ground_truths.append(category_name)
                
                # COCO的bbox格式是 [x, y, width, height]
                # 转换为 [x1, y1, x2, y2] 格式
                x, y, w, h = ann['bbox']
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                bboxes.append([x1, y1, x2, y2])

            # 构建符合要求的字典结构
            data_entry = {
                "image_id": image_path,
                "ground_truth": ground_truths,
                "bbox": bboxes
            }
            
            # 写入JSONL文件
            f_out.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

    print(f"处理完成！")
    print(f"原始图片数量: {len(images_info)}")
    print(f"原始标注数量: {len(annotations_info)}")
    print(f"JSONL文件已保存到: {output_file}")



dataset_base_path = '/home/xray/xray_common/pidray/pidray'  # 例如: '/home/xray/xray_common/NewDataset'
output_jsonl_file = '/home/data2/zkj/llt_code/STING-BEE/dataset/pidray/trainset_all.jsonl'

convert_coco_to_jsonl(dataset_base_path, output_jsonl_file)
