import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics for fake news classification')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth CSV file')
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to the predictions CSV file')
    parser.add_argument('--column_name', type=str, required=True, help='Column name for the predictions in the CSV file')
    args = parser.parse_args()

    # Load the datasets
    print("ZERO SHOT")
    train_df_gt = pd.read_csv(args.ground_truth)
    train_df_predictions = pd.read_csv(args.prediction_file)

    # Extract the columns of interest
    y_gt_filtered = train_df_gt['majority_vote']
    y_pred_filtered = train_df_predictions[args.column_name]

    print(len(y_gt_filtered), len(y_pred_filtered))

    # Calculate metrics
    accuracy = accuracy_score(y_gt_filtered, y_pred_filtered)
    precision = precision_score(y_gt_filtered, y_pred_filtered, average='weighted')
    recall = recall_score(y_gt_filtered, y_pred_filtered, average='weighted')
    f1 = f1_score(y_gt_filtered, y_pred_filtered, average='weighted')
    
    # Print results
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')