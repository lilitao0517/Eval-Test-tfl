import json
import argparse
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def normalize_label_strict(answer):
    if not isinstance(answer, str):
        return False, None
    letters = [ch.upper() for ch in answer if 'A' <= ch.upper() <= 'Z']
    if len(letters) != 1:
        return False, None
    if len(answer.strip()) > 3:
        return False, None
    return True, letters[0]

# def normalize_label_strict(answer):
#     if not isinstance(answer, str):
#         return False, None
#     # 遍历字符串，找到第一个英文字母（A-Z 或 a-z）
#     for ch in answer:
#         if 'A' <= ch.upper() <= 'Z':
#             return True, ch.upper()
#     # 如果没有找到任何字母
#     return False, None



def load_labels(predictions_file):
    y_true = []
    y_pred = []
    skipped = 0
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                gt = data.get("ground_truth", "")
                pred = data.get("answer", "")
                
                gt_valid, gt_label = normalize_label_strict(gt)
                pred_valid, pred_label = normalize_label_strict(pred)
                
                if gt_valid and pred_valid:
                    y_true.append(gt_label)
                    y_pred.append(pred_label)
                else:
                    skipped += 1
            except:
                skipped += 1
                continue
    print(f"Total skipped: {skipped}")
    return y_true, y_pred

def compute_metrics_and_save(y_true, y_pred, output_txt):
    all_classes = sorted(set(y_true + y_pred))
    print(all_classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    report_dict = classification_report(
        y_true, y_pred, labels=all_classes, 
        output_dict=True, zero_division=0
    )
    # print(report_dict)
    # raise Exception("debug")
    micro_precision = report_dict['weighted avg']['precision']
    micro_recall = report_dict['weighted avg']['recall']
    micro_f1 = report_dict['weighted avg']['f1-score']
    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']
    mAP = micro_f1
    
    with open(output_txt, 'w') as f:
        f.write("=== Classification Evaluation Results ===\n\n")
        
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Classes: {all_classes}\n\n")
        f.write(f"weighted avg-Precision: {micro_precision:.4f}\n")
        f.write(f"weighted avg-Recall:    {micro_recall:.4f}\n")
        f.write(f"weighted avg-F1:        {micro_f1:.4f}\n")
        f.write(f"Macro-Precision: {macro_precision:.4f}\n")
        f.write(f"Macro-Recall:    {macro_recall:.4f}\n")
        f.write(f"Macro-F1:        {macro_f1:.4f}\n")
        f.write(f"mAP (Accuracy):  {mAP:.4f}\n\n")
        
        f.write("Per-class metrics:\n")
        f.write("{:<8} {:<12} {:<12} {:<12}\n".format("Class", "Precision", "Recall", "F1-Score"))
        f.write("-" * 45 + "\n")
        for cls in all_classes:
            p = report_dict[cls]['precision']
            r = report_dict[cls]['recall']
            f1 = report_dict[cls]['f1-score']
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(cls, p, r, f1))
        f.write("\n")
        
        f.write("Confusion Matrix (rows: true, cols: pred):\n")
        f.write("     " + " ".join([f"{c:>6}" for c in all_classes]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{all_classes[i]:<4} " + " ".join([f"{x:>6}" for x in row]) + "\n")
    
    print(f"Evaluation results saved to: {output_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to JSONL file with 'answer' and 'ground_truth'")
    parser.add_argument("--output", type=str, default="evaluation_results.txt", help="Output .txt file path")
    args = parser.parse_args()
    
    if not os.path.isfile(args.predictions):
        print(f"Error: File {args.predictions} not found.")
        return
    
    y_true, y_pred = load_labels(args.predictions)
    
    if len(y_true) == 0:
        print("Error: No valid samples found.")
        return
        
    compute_metrics_and_save(y_true, y_pred, args.output)

if __name__ == "__main__":
    main()