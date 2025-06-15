import pandas as pd
import torch
from transformers import pipeline
import asyncio
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

MODEL_NAME = "facebook/bart-large-cnn"

# Function to summarize text
def summarize_text(text, max_length=200, summary_length=150, do_sample=False):
    # Check if the text length is more than max_length
    if len(text.split()) < max_length:
        print("RETURNING ORIGINAL TEXT")
        return text
    else:
        summary = summarizer(text, 
                             max_length=max_length, 
                             min_length=summary_length, 
                             do_sample=do_sample)
        return summary[0]['summary_text']

async def summarize_texts_async(texts):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        results = await asyncio.gather(
            *[loop.run_in_executor(pool, summarize_text, text) for text in texts]
        )
    return results

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Parallel summarization of text data using BART.")
    parser.add_argument('--data_file', type=str, default='filtered_unique_batch3.csv',
                        help='Path to the input CSV file containing text data.')
    parser.add_argument('--output_file', type=str, default='batch_bad2.csv',
                        help='Path to the output CSV file where summaries will be saved.')
    args = parser.parse_args()


    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    print("Using GPU" if device == 0 else "Using CPU")

    # Initialize the summarization pipeline with the BART model
    summarizer = pipeline("summarization", 
                          model=MODEL_NAME, 
                          device=device)

    # Read the DataFrame
    df = pd.read_csv(args.data_file)  # Read the entire CSV file

    # Create an event loop
    loop = asyncio.get_event_loop()

    # Process data in batches for efficiency
    batch_size = 8  # Adjust batch size based on your GPU memory and performance
    text_batches = [df['text_content'][i:i+batch_size].tolist() for i in range(0, len(df), batch_size)]

    output_filename = args.output_file

    # Create the output file with headers if it does not exist
    if not os.path.exists(output_filename):
        pd.DataFrame(columns=['unique_id', 
                              'title', 
                              'text_content', 
                              'text_content_summary']).to_csv(output_filename, index=False)

    # Asynchronously process each batch with a progress bar
    for batch_index, batch in enumerate(tqdm(text_batches, 
                                             desc="Summarizing batches")):
        summarized_batch = loop.run_until_complete(summarize_texts_async(batch))
        
        # Create DataFrame for the current batch
        batch_df = pd.DataFrame({
            'unique_id': df['unique_id'][batch_index*batch_size:(batch_index+1)*batch_size].tolist(),
            'title': df['title'][batch_index*batch_size:(batch_index+1)*batch_size].tolist(),
            'text_content': batch,
            'text_content_summary': summarized_batch
        })
        
        # Append the current batch results to the CSV file
        batch_df.to_csv(output_filename, mode='a', header=False, index=False)

    print(f"Summarization complete. Data saved incrementally to '{output_filename}'")
