import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# This script splits a dataset into training, validation, and test sets.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument('--input_file', type=str, default='Combined_labelled_batch1_2_3.csv',
                        help='Path to the input CSV file containing the dataset.')
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.input_file)

    # Filter rows with 'Real' and 'Fake' labels
    real_rows = df[df['label'] == 'Real']
    fake_rows = df[df['label'] == 'Fake']

    # Sample 4000 rows from each
    real_sampled = real_rows.sample(n=4000, random_state=42)
    fake_sampled = fake_rows.sample(n=4000, random_state=42)

    # Combine the sampled rows
    combined_sampled = pd.concat([real_sampled, fake_sampled])

    # Select the required columns
    selected_columns = ['unique_id', 
                        'title', 
                        'text_content_summary', 
                        'label', 
                        'explanation']
    final_df = combined_sampled[selected_columns]

    # Split the data into 70% training, 15% validation, and 15% test sets
    train_df, temp_df = train_test_split(final_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save the new CSV files
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
