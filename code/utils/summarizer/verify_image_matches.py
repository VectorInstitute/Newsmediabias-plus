import os
import logging
import argparse
import pandas as pd

# # This script checks if unique IDs in a dataset match image files in a specified directory.

def setup_logging(log_file_path):
    """Configure logging."""
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def load_dataframe(csv_path):
    """Load dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    return pd.read_csv(csv_path)

def find_matches(df, source_dir):
    """Check for matching image files based on unique_id.
    Args:
        df (pd.DataFrame): DataFrame containing unique IDs.
        source_dir (str): Directory where image files are stored.
    Returns:
        list: List of unmatched unique IDs.
    """
    unmatched = []
    all_files = os.listdir(source_dir)

    for unique_id in df['unique_id'].unique():
        matched = any(str(unique_id) in file for file in all_files)
        if matched:
            matches = [file for file in all_files if str(unique_id) in file]
            for match in matches:
                logging.info(f"Match found: {match} for unique ID {unique_id}")
        else:
            unmatched.append(unique_id)
            logging.warning(f"No match found for unique ID {unique_id}")

    return unmatched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match unique IDs in a dataset to image files in a directory.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV file (example: balanced_filtered_dataset.csv)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory where image files are stored")
    parser.add_argument("--log_file", type=str, default="match_log.txt", help="Path to save the log file")

    args = parser.parse_args()

    setup_logging(args.log_file)

    df = load_dataframe(args.csv_path)
    unmatched = find_matches(df, args.image_dir)

    if unmatched:
        logging.warning(f"Total unmatched unique IDs: {len(unmatched)}")
        logging.warning(f"Unmatched list: {unmatched}")
        print(f"Finished with {len(unmatched)} unmatched IDs. Check log for details.")
    else:
        logging.info("All unique IDs have corresponding image files.")
        print("All unique IDs matched successfully. See log for details.")