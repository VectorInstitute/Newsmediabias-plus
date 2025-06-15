
import pandas as pd
import time
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"



# Function to truncate content to 1000 words
def truncate_to_5000_words(text):
    # Remove extra spaces and new lines
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    return ' '.join(words[:5000]) if len(words) > 5000 else cleaned_text

# Define few-shot examples
few_shot_examples = [
    {
        "role": "user",
        "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
    },
    {
        "role": "assistant",
        "content": "Article: 'The new policy is just another attempt by the government to control the population. It's clear that they don't care about our freedoms.'\nClassification: Heavily Biased<|LABEL_SEP|>\nBiased words/phrases: 'attempt by the government to control'<|SEPARATOR|>'they don't care about our freedoms'"
    },
    {
        "role": "user",
        "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
    },
    {
        "role": "assistant",
        "content": "Article: 'The recent economic policy changes are expected to improve the economy. Experts believe this will lead to a positive outcome for the country.'\nClassification: Unbiased<|LABEL_SEP|>\nBiased words/phrases: None"
    },
    {
        "role": "user",
        "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
    },
    {
        "role": "assistant",
        "content": "Article: 'The opposition party criticized the policy as a half-baked measure that would do more harm than good.'\nClassification: Moderately Biased<|LABEL_SEP|>\nBiased words/phrases: 'half-baked measure'<|SEPARATOR|>'more harm than good'"
    },
    {
        "role": "user",
        "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
    },
    {
        "role": "assistant",
        "content": "Article: 'The proposal has its critics, but it also has some merits that should not be overlooked.'\nClassification: Slightly Biased<|LABEL_SEP|>\nBiased words/phrases: 'has its critics'<|SEPARATOR|>'should not be overlooked'"
    },
    {
        "role": "user",
        "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
    },
    {
        "role": "assistant",
        "content": "Article: 'The government announced new measures to tackle climate change, aiming for a significant reduction in emissions by 2030.'\nClassification: Unbiased<|LABEL_SEP|>\nBiased words/phrases: None"
    }
]

# Function to detect bias in an article
def detect_bias(article, model, tokenizer):
    """
    Detects bias in a given article using the Llama-3 model.
    """
    print(f"Processing: {article[:30]}...")  # Print first 30 characters of the article
    prompt = few_shot_examples + [
        {"role": "user", "content": "You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list the biased words found in the article, separated by '<|SEPARATOR|>'."},
        {"role": "assistant", "content": article},
    ]

    input_text = "".join([f"{item['role']}: {item['content']}\n" for item in prompt])
    
    input_ids = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  # Adjust this value if necessary
    ).to(model.device)

    eos_token_id = tokenizer.eos_token_id

    try:
        outputs = model.generate(
            input_ids['input_ids'],
            max_new_tokens=200,  # Increase the max_new_tokens to allow space for reasoning
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = outputs[0][input_ids['input_ids'].shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        print(f"Completed: {decoded_response}")  # Print the response
        
        # Extracting the label and biased words/phrases from the response
        label = "Unbiased"  # Default label
        biased_words = []

        if "<|LABEL_SEP|>" in decoded_response:
            parts = decoded_response.split("<|LABEL_SEP|>")
            label_section = parts[0]
            biased_words_section = parts[1] if len(parts) > 1 else ""

            if "heavily biased" in label_section.lower():
                label = "Heavily Biased"
            elif "moderately biased" in label_section.lower():
                label = "Moderately Biased"
            elif "slightly biased" in label_section.lower():
                label = "Slightly Biased"
            
            if "Biased words/phrases:" in biased_words_section:
                biased_words_section = biased_words_section.split("Biased words/phrases:")[1]
                biased_words = biased_words_section.strip().split("<|SEPARATOR|>")

        return label, biased_words
    except Exception as e:
        return f"Error processing article: {str(e)}", []

# Function to process a batch of data
def process_batch(data, model, tokenizer):
    """
    Processes a batch of articles to detect bias.
    """
    results = []
    for _, row in data.iterrows():
        article = truncate_to_5000_words(row['text_content'])
        label, biased_words = detect_bias(article, model, tokenizer)
        results.append((row['_id'], row['title'], row['canonical_link'], row['first_paragraph'], row['outlet'],
                        row['source_url'], row['topics'], row['unique_id'], row['date_published'], article,
                        label, biased_words))
    return results

# Function to save results to CSV
def save_results(results, output_file='bulk-annotations-hf2.csv'):
    """
    Saves the results to a CSV file.
    """
    results_df = pd.DataFrame(results, columns=['_id', 'title', 'canonical_link', 'first_paragraph', 'outlet',
       'source_url', 'topics', 'unique_id', 'date_published', 'article', 'llama_label', 'biased_words'])
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Appended {len(results)} records to {output_file}")
    else:
        results_df.to_csv(output_file, mode='w', header=True, index=False)
        print(f"Created and wrote {len(results)} records to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process news articles to detect bias using Llama-3 model.")
    parser.add_argument('--data_path', type=str, default='nmb-plus-texts_selected_part1.csv',
                        help='Path to the input CSV file containing news articles.')
    parser.add_argument('--output_file', type=str, default='bulk-annotations-hf2.csv',
                        help='Path to the output CSV file for saving results.')
    args = parser.parse_args()


    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token correctly


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
    batch_size = 25
    num_batches = (len(data) + batch_size - 1) // batch_size

    start_time = time.time()
    for i in range(num_batches):
        batch_data = data.iloc[i * batch_size:(i + 1) * batch_size]
        if not batch_data.empty:
            results = process_batch(batch_data, model, tokenizer)
            save_results(results, output_file=args.output_file)
            print(f"Batch {i+1}/{num_batches} processed and saved.")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            print(f"Total time taken so far: {int(hours)} hours and {int(minutes)} minutes")
            
            time.sleep(4)

    print("All batches processed successfully.")