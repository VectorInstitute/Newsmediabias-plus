import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def split_data(input_csv, output_train_csv, 
               output_val_csv, output_test_csv, 
               train_size=0.8, val_size=0.1, test_size=0.1):
    df = pd.read_csv(input_csv)
    
    train_df, temp_df = train_test_split(df, 
                                         train_size=train_size, 
                                         random_state=42)
    
    val_relative_size = val_size / (val_size + test_size)
    
    val_df, test_df = train_test_split(temp_df, 
                                       train_size=val_relative_size, 
                                       random_state=42)
    
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)

    print(f"Data split complete! Train: {len(train_df)}, \
          Val: {len(val_df)}, Test: {len(test_df)}")


def create_two_datasets(csv_path, output_path_1, 
                        output_path_2, sample_size_1=200, 
                        sample_size_2=200):
    df = pd.read_csv(csv_path)
    
    if 'content' not in df.columns or 'text_label' not in df.columns:
        raise ValueError("The CSV must contain 'content' and 'text_label' columns")
    
    likely_samples = df[df['text_label'] == 'Likely'].sample(frac=1, 
                                                             random_state=42).reset_index(drop=True)
    unlikely_samples = df[df['text_label'] == 'Unlikely'].sample(frac=1, 
                                                                 random_state=42).reset_index(drop=True)
    
    total_required = sample_size_1 // 2 + sample_size_2 // 2
    if len(likely_samples) < total_required or len(unlikely_samples) < total_required:
        raise ValueError(f"Not enough samples for each label. Likely: {len(likely_samples)}, \
                         Unlikely: {len(unlikely_samples)}")
    
    likely_sample_1 = likely_samples.iloc[:sample_size_1 // 2]
    likely_sample_2 = likely_samples.iloc[sample_size_1 // 2:sample_size_1 // 2 + sample_size_2 // 2]
    
    unlikely_sample_1 = unlikely_samples.iloc[:sample_size_1 // 2]
    unlikely_sample_2 = unlikely_samples.iloc[sample_size_1 // 2:sample_size_1 // 2 + sample_size_2 // 2]
    
    dataset_1 = pd.concat([likely_sample_1, unlikely_sample_1]).sample(frac=1, 
                                                                       random_state=42).reset_index(drop=True)
    dataset_2 = pd.concat([likely_sample_2, unlikely_sample_2]).sample(frac=1, 
                                                                       random_state=42).reset_index(drop=True)
    
    dataset_1.to_csv(output_path_1, index=False)
    dataset_2.to_csv(output_path_2, index=False)
    
    print(f"Created two balanced datasets:")
    print(f"- Dataset 1: {sample_size_1} instances saved to {output_path_1}")
    print(f"  Likely instances: {len(dataset_1[dataset_1['text_label'] == 'Likely'])}")
    print(f"  Unlikely instances: {len(dataset_1[dataset_1['text_label'] == 'Unlikely'])}")
    print(f"- Dataset 2: {sample_size_2} instances saved to {output_path_2}")
    print(f"  Likely instances: {len(dataset_2[dataset_2['text_label'] == 'Likely'])}")
    print(f"  Unlikely instances: {len(dataset_2[dataset_2['text_label'] == 'Unlikely'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument("--input_csv", type=str, default="Benchmarking/dataset.csv", 
                        help="Path to the input CSV file.")
    parser.add_argument("--output_train_csv", type=str, default="Benchmarking/sample_train.csv", 
                        help="Path to the output training CSV file.")
    parser.add_argument("--output_test_csv", type=str, default="Benchmarking/sample_test.csv", 
                        help="Path to the output test CSV file.")
    parser.add_argument("--sample_size_1", type=int, default=10000,
                        help="Sample size for the first dataset.")
    parser.add_argument("--sample_size_2", type=int, default=11000,
                        help="Sample size for the second dataset.")
    args = parser.parse_args()
    create_two_datasets(args.input_csv, 
                        args.output_train_csv, 
                        args.output_test_csv, 
                        args.sample_size_1, 
                        args.sample_size_2)