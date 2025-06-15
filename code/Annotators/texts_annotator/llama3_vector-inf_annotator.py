import pandas as pd
import time
import os
import json
from openai import OpenAI
import asyncio
import sys
import argparse
from prompt import one_shot, two_shot, three_shot, four_shot, five_shot

MODEL_NAME = "Meta-Llama-3-8B-Instruct" # or "Mistral-7B-Instruct-v0.3"

# Function to truncate content to 5000 words
def truncate_to_5000_words(text):
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    return ' '.join(words[:5000]) if len(words) > 5000 else cleaned_text

def detect_bias_sync(article, model):
    """
    Synchronous function to detect bias in an article using the OpenAI API.
    """
    print(f"Processing: {article[:30]}...")

    completion = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{one_shot}"},
            {"role": "assistant", "content": "\nClassification: Heavily Biased \nBiased words/phrases: 'attempt by the government to control', 'they don't care about our freedoms'"},
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{two_shot}"},
            {"role": "assistant", "content": "\nClassification: Unbiased \nBiased words/phrases: None"},
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{three_shot}"},
            {"role": "assistant", "content": "\nClassification: Moderately Biased \nBiased words/phrases: 'half-baked measure', 'more harm than good'"},
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{four_shot}"},
            {"role": "assistant", "content": "\nClassification: Slightly Biased \nBiased words/phrases: 'has its critics', 'should not be overlooked'"},
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{five_shot}"},
            {"role": "assistant", "content": "\nClassification: Unbiased \nBiased words/phrases: None"},
            {"role": "user", "content": f"You are a helpful news media bias detection bot. Classify the article as 'Unbiased', 'Slightly Biased', 'Moderately Biased', or 'Heavily Biased'. List biased words if any found separated by '':{article}"}
        ]
    )

    response = completion.choices[0].message.content
    
    # Extracting the label and biased words/phrases from the response
    label = "Unbiased" # Default label
    biased_words = []

    if "Biased words/phrases:" in response:
        parts = response.split("Biased words/phrases:")
        label_section = parts[0]
        biased_words_section = parts[1] if len(parts) > 1 else ""

        if "heavily biased" in label_section.lower():
            label = "Heavily Biased"
        elif "moderately biased" in label_section.lower():
            label = "Moderately Biased"
        elif "slightly biased" in label_section.lower():
            label = "Slightly Biased"

        biased_words = [word.strip() for word in biased_words_section.split(',') if word.strip()]

    print("LABEL:", label, "BIASED WORDS:", biased_words)
    return label, biased_words

async def detect_bias(article, model):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, detect_bias_sync, article, model)

# Function to process a batch of data
async def create_completion(row):
    """
    Process a single row of data to detect bias in the article text.
    """
    article = truncate_to_5000_words(row['text_content'])
    label, biased_words = await detect_bias(article, MODEL_NAME)
    json_result = {
        "data": {
            "text_content": row['text_content'],
            "ref_id": row['unique_id'],
            "meta_info": {
                "timestamp": row['date_published'],
                "location": row['outlet']
            }
        },
        "predictions": [{
            "result": [{
                "from_name": "bias_classification",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [label]},
                "biased_words": biased_words
            }]
        }]
    }

    result = [
        row['title'], row['canonical_link'], row['first_paragraph'],
        row['outlet'], row['source_url'], row['topics'], row['unique_id'],
        row['date_published'], article, label, biased_words
    ]
    return result, json_result

# Function to send multiple requests
async def send_requests(data_batch):
    tasks = [create_completion(row) for _, row in data_batch.iterrows()]
    return await asyncio.gather(*tasks)

# Function to save results to CSV
def save_results(results_csv, results_json, output_file='batch_bulk-annotations-vl-bad.csv', json_file='batch-bad_mistral.v5_json.json'):
    """
    Save results to CSV and JSON files.
    """
    results_df = pd.DataFrame(results_csv, columns=[
       'title', 'canonical_link', 'first_paragraph', 'outlet',
        'source_url', 'topics', 'unique_id', 'date_published', 'article',
        'llama_label', 'biased_words'
    ])
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file, mode='w', header=True, index=False)

    file_path = json_file
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            json.dump(results_json, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump(results_json, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and annotate text data with LLM.")
    parser.add_argument('--model', type=str, default='Meta-Llama-3-8B-Instruct', help='Model to use for inference')
    parser.add_argument('--data_path', type=str, default='us_elections_combined_data_n5.csv', help='Path to the input data CSV file')
    parser.add_argument('--output_file', type=str, default='batch_bulk-annotations-vl-bad.csv', help='Path to save the output CSV file')
    parser.add_argument('--json_file', type=str, default='batch-bad_mistral.v5_json.json', help='Path to save the output JSON file')
    args = parser.parse_args()


    # Initialize OpenAI client
    client = OpenAI(base_url="http://gpu034:8080/v1", api_key="EMPTY")

    # Load data
    data_path = args.data_path
    data = pd.read_csv(data_path)

    # Determine where to resume processing
    processed_ids = set()
    if os.path.exists(args.output_file):
        processed_data = pd.read_csv(args.output_file)
        processed_ids = set(processed_data['unique_id'])

    data = data[~data['unique_id'].isin(processed_ids)]

    # Batch processing
    batch_size = 15
    num_batches = (len(data) + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(num_batches):
        prompts = data.iloc[i * batch_size:(i + 1) * batch_size]
        if not prompts.empty:
            results = asyncio.run(send_requests(prompts))
            results_csv, results_json = zip(*results)
            save_results(results_csv, results_json, output_file=args.output_file, json_file=args.json_file)
            print(f"Batch {i + 1}/{num_batches} processed and saved.")

            end_time = time.time()
            elapsed_time = end_time - start_time
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            seconds = elapsed_time % 60
            print(f"Total time taken so far: hours: {hours}, minutes: {minutes}, seconds: {seconds}")

            time.sleep(4)  # Sleep for 4 seconds after each batch

    print("All batches processed successfully.")