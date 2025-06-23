import argparse
import pandas as pd

# This script compares multiple CSV files containing labels and explanations, calculates the majority vote for each unique ID, and outputs a new CSV file with the results.

def majority_vote(labels, explanations):
    """Calculate the majority vote from a list of labels and explanations."""
    mode_label = labels.mode()[0]
    return mode_label

def compare_csvs(csv_paths, label_columns, explanation_columns, output_csv_path):
    """Compare CSV files and output a combined CSV with majority vote labels."""
    dataframes = []
    
    # Read in each CSV file and store it in a list
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dataframes.append(df)
    
    # Check if the 'unique_id' columns match across all CSV files
    unique_ids = [df['unique_id'] for df in dataframes]
    if not all(unique_ids[0].equals(ids) for ids in unique_ids):
        raise ValueError("The 'unique_id' columns do not match across the CSV files.")
    
    # Create the combined dataframe
    combined_df = pd.DataFrame({'unique_id': dataframes[0]['unique_id']})
    
    # Add title and text_content_summary from the first dataframe (assuming they are the same across CSVs)
    combined_df['title'] = dataframes[0]['title']
    combined_df['text_content_summary'] = dataframes[0]['text_content_summary']
    
    # Add label and explanation columns for each CSV
    for i, df in enumerate(dataframes):
        label_column = label_columns[i]
        explanation_column = explanation_columns[i]
        combined_df[f'label_{i+1}_summary'] = df[label_column]
        combined_df[f'explanation_{i+1}_summary'] = df[explanation_column]
    
    # Apply majority vote to determine the final label
    combined_df['majority_vote'] = combined_df.apply(
        lambda row: majority_vote(
            row[[f'label_{i+1}_summary' for i in range(len(csv_paths))]],
            row[[f'explanation_{i+1}_summary' for i in range(len(csv_paths))]]
        ), axis=1
    )
    
    # Create the output dataframe with just the relevant columns
    output_df = combined_df[['unique_id', 'title', 'text_content_summary', 'majority_vote']]
    
    # Write the result to a new CSV file
    output_df.to_csv(output_csv_path, index=False)

def main():
    # Set up argparse to take CSV files, column names for labels, and column names for explanations
    parser = argparse.ArgumentParser(description="Compare multiple CSVs and calculate the majority vote.")
    parser.add_argument('csv_files', metavar='csv', type=str, nargs='+', 
                        help="Paths to the CSV files to be compared.")
    parser.add_argument('output_csv', type=str, 
                        help="Path to the output CSV file.")
    parser.add_argument('--label_columns', metavar='label', type=str, nargs='+',
                        help="Column names containing the labels for each CSV file.")
    parser.add_argument('--explanation_columns', metavar='explanation', type=str, nargs='+',
                        help="Column names containing the explanations for each CSV file.")
    
    args = parser.parse_args()

    # Ensure that the number of label columns and explanation columns matches the number of CSV files
    if len(args.label_columns) != len(args.csv_files):
        raise ValueError("The number of label columns must match the number of CSV files.")
    
    if len(args.explanation_columns) != len(args.csv_files):
        raise ValueError("The number of explanation columns must match the number of CSV files.")
    
    # Call the compare_csvs function with the arguments provided
    compare_csvs(args.csv_files, args.label_columns, args.explanation_columns, args.output_csv)

if __name__ == "__main__":
    main()
