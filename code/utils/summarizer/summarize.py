# Import libraries
import pandas as pd
from transformers import pipeline
from argparse import ArgumentParser

# Function to summarize text
def summarize_text(text, max_length=200, summary_length=150, do_sample=False):
    """
    Summarizes the input text using a pre-trained model.
    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the input text in tokens.
        summary_length (int): Desired length of the summary in tokens.
        do_sample (bool): Whether to use sampling for the summary generation.
    Returns:
        str: The summarized text.
    """
    # Check if the text length is more than max_length
    if len(text.split()) > max_length:
        # Truncate the text to max_length tokens
        text = ' '.join(text.split()[:max_length])
    summary = summarizer(text, max_length=max_length, min_length=summary_length, do_sample=do_sample)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Initialize the summarization pipeline with the BART model

    parser = ArgumentParser(description="Summarize text_content column in a CSV using a summarization model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, default="summarized_data.csv", help="Path to save the output CSV")
    parser.add_argument("--sample_n", type=int, default=10, help="Number of rows to sample for summarization")
    args = parser.parse_args()

    summarizer = pipeline("summarization", 
                          model="facebook/bart-large-cnn")

    # Read the DataFrame
    df = pd.read_csv(args.input_csv).sample(args.sample_n)  # Replace with your actual file path ('balanced_filtered_dataset.csv' is an example)

    # Apply summarization to the 'text_content' column
    df['text_content_summary'] = df['text_content'].apply(lambda x: summarize_text(x))

    # Save the new DataFrame with the added summary column
    df.to_csv(args.output_csv,
              index=False)

    print("Summarization complete. New DataFrame saved as ", args.output_csv)


# To run this script, use the following command in the terminal:
# python summarize.py \
# --input_csv your_input_file.csv \
# --output_csv your_output_file.csv \
# --sample_n 10