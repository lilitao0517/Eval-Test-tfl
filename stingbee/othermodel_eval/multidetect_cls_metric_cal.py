import json
import argparse
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_multilabel_answer(answer):
    """
    从任意字符串中提取 A-Z 选项字母，去重、排序、转大写。
    返回列表，如 ["A", "C"]
    """
    if not isinstance(answer, str):
        return []
    letters = [ch.upper() for ch in answer if 'A' <= ch.upper() <= 'Z']
    return sorted(set(letters))

def load_labels(predictions_file):
    """加载所有样本的 ground_truth 和 answer，解析为标签列表"""
    y_true = []
    y_pred = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                gt = data.get("ground_truth", "")
                pred = data.get("answer", "")
                y_true.append(parse_multilabel_answer(gt))
                y_pred.append(parse_multilabel_answer(pred))
            except:
                y_true.append([])
                y_pred.append([])
    return y_true, y_pred

def compute_multilabel_metrics(y_true, y_pred, all_classes):
    """计算 multi-label 指标"""
    if not all_classes:
        raise ValueError("No classes found!")
    
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    n_samples = len(y_true)
    n_classes = len(all_classes)
    y_true_bin = np.zeros((n_samples, n_classes), dtype=int)
    y_pred_bin = np.zeros((n_samples, n_classes), dtype=int)
    
    for i, (true_labels, pred_labels) in enumerate(zip(y_true, y_pred)):
        for label in true_labels:
            if label in class_to_idx:
                y_true_bin[i, class_to_idx[label]] = 1
        for label in pred_labels:
            if label in class_to_idx:
                y_pred_bin[i, class_to_idx[label]] = 1
    micro_p = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    micro_r = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    
    macro_p = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_r = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    
    return {
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f1,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'mAP': micro_f1, 
        'y_true_bin': y_true_bin,
        'y_pred_bin': y_pred_bin
    }

def save_results(metrics, all_classes, total_samples, output_file):
    """保存评估结果到文本文件"""
    with open(output_file, 'w') as f:
        f.write("=== Multi-label Classification Evaluation Results ===\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Classes: {all_classes}\n\n")
        
        f.write(f"Micro-Precision: {metrics['micro_precision']:.4f}\n")
        f.write(f"Micro-Recall:    {metrics['micro_recall']:.4f}\n")
        f.write(f"Micro-F1:        {metrics['micro_f1']:.4f}\n")
        f.write(f"Macro-Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro-Recall:    {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro-F1:        {metrics['macro_f1']:.4f}\n")
        f.write(f"mAP (Micro-F1):  {metrics['mAP']:.4f}\n")
    
    print(f"Evaluation results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-label classification results.")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to JSONL file with 'answer' and 'ground_truth'")
    parser.add_argument("--output", type=str, default="multilabel_evaluation.txt",
                        help="Output .txt file path")
    args = parser.parse_args()
    
    if not os.path.isfile(args.predictions):
        print(f"Error: File {args.predictions} not found.")
        return
    y_true, y_pred = load_labels(args.predictions)
    total_samples = len(y_true)
    
    if total_samples == 0:
        print("Error: No samples loaded.")
        return
    
    all_labels = set()
    for labels in y_true + y_pred:
        all_labels.update(labels)
    all_classes = sorted(all_labels)
    
    if not all_classes:
        print("Warning: No valid classes found in data.")
        with open(args.output, 'w') as f:
            f.write("No valid classes found.\n")
        return
    
    metrics = compute_multilabel_metrics(y_true, y_pred, all_classes)
    save_results(metrics, all_classes, total_samples, args.output)

if __name__ == "__main__":
    main()