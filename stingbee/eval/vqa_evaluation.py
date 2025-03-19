import pandas as pd
import json
import argparse

def compute_accuracy(predictions_file, groundtruth_file, questions_file, category=None):
    """
    Computes accuracy based on model predictions and ground truth answers.

    Args:
        predictions_file (str): Path to the JSONL file containing model predictions.
        groundtruth_file (str): Path to the Excel file containing ground truth answers.
        questions_file (str): Path to the JSONL file containing questions.
        category (str, optional): If specified, computes accuracy only for this category.

    Returns:
        float: Accuracy as a percentage.
    """
    # Load the ground truth from Excel
    groundtruth_df = pd.read_excel(groundtruth_file)

    # Convert the ground truth DataFrame to a dictionary with question_id as the key
    groundtruth_dict = groundtruth_df.set_index('question_id')['correct_answer'].to_dict()

    # Load questions to get category information
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    # Create a dictionary to map question_id to category
    question_category_map = {q['question_id']: q['category'] for q in questions}

    # If category filtering is needed, adjust the ground truth dictionary accordingly
    if category is not None:
        groundtruth_dict = {q_id: ans for q_id, ans in groundtruth_dict.items()
                            if question_category_map.get(q_id) == category}

    # Load predictions from JSONL file
    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line) for line in f]

    # Track correct and total answers
    correct = 0
    total = 0

    for prediction in predictions:
        question_id = prediction['question_id']
        predicted_answer = prediction['answer']

        # Skip if the question_id is not in the filtered ground truth
        if question_id not in groundtruth_dict:
            continue

        # Compare the predicted answer to the ground truth
        correct_answer = groundtruth_dict[question_id]
        if predicted_answer.strip().lower() == correct_answer.strip().lower():
            correct += 1

        total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total}) for category: {category if category else 'Overall'})")

    return accuracy

def main():
    """
    Main function to parse command-line arguments and compute accuracy.
    """
    parser = argparse.ArgumentParser(description="Compute accuracy for VQA model predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the model predictions JSONL file.")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the ground truth Excel file.")
    parser.add_argument("--questions", type=str, required=True, help="Path to the questions JSONL file.")
    parser.add_argument("--category", type=str, default=None, help="Optional category to filter results.")

    args = parser.parse_args()

    # Compute accuracy
    compute_accuracy(args.predictions, args.groundtruth, args.questions, args.category)

if __name__ == "__main__":
    main()
