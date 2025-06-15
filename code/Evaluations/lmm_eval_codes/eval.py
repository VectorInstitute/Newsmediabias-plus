import json
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth labels.")
    parser.add_argument("--results_file", type=str, help="Path to the JSON file containing model predictions and ground truth labels.")
    args = parser.parse_args()

    # Load data from JSON file
    with open(args.results_file, "r") as file:
        data = json.load(file)

    # Preprocess the predicted and ground truth labels
    predicted_labels = [item["predicted_answer"].replace("Classification: ", "").strip() for item in data]
    ground_truth_labels = [item["ground_truth"].strip() for item in data]

    # Calculate metrics
    precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
    recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')
    f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
