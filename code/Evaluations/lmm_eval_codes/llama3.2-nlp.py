import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Features, Value
from argparse import ArgumentParser
import json
from tqdm import tqdm
import time

# Define dataset features
features = Features({
    "unique_id": Value("string"),
    "outlet": Value("string"),
    "headline": Value("string"),
    "article_text": Value("string"),
    "date_published": Value("string"),
    "source_url": Value("string"),
    "canonical_link": Value("string"),
    "new_categories": Value("string"),
    "news_categories_confidence_scores": Value("string"),
    "text_label": Value("string"),
    "multimodal_label": Value("string"),
})

# Load dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Dataset size: {len(dataset)} entries")
    return dataset

# Truncate content
def truncate_to_2000_words(text):
    words = text.split()
    return ' '.join(words[:2000]) if len(words) > 2000 else text

# Collate function for DataLoader
def collate_fn(batch):
    return {
        'article_texts': [item['article_text'] for item in batch],
        'labels': [item['text_label'] for item in batch],
        'ids': [item['unique_id'] for item in batch]
    }

# Evaluation function
def evaluate(model, dataloader, tokenizer, device, save_path):
    model.eval()
    results = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            for batch in dataloader:
                texts, labels, ids = (
                    batch['article_texts'], batch['labels'], batch['ids']
                )
                batch_results = []  # Store results for the current batch
                for text, label, id in zip(texts, labels, ids):
                    try:
                        text = truncate_to_2000_words(text)
                        prompt = (
                            "Please analyze the following article and determine whether it contains any signs of disinformation. "
                            "If you find the content to be misleading, biased, or negative, respond with 'Classification: Likely.' "
                            "If the content seems accurate, credible, and unbiased, respond with 'Classification: Unlikely.' "
                            "Please choose only one classification.\n\n"
                            f"Article: {text}\n\n"
                        )


                        # Prepare inputs
                        inputs = tokenizer(prompt, 
                                           return_tensors="pt", 
                                           truncation=True, 
                                           padding=True).to(device)

                        # Generate output
                        output = model.generate(
                            **inputs, 
                            max_new_tokens=50, 
                            temperature=0.7, 
                            top_p=0.9, 
                            repetition_penalty=1.2
                        )

                        predicted_answer = tokenizer.decode(output[0], skip_special_tokens=True)

                        # Clean and extract the predicted answer
                        if "Classification: Likely" in predicted_answer:
                            predicted_answer = "Likely"
                        elif "Classification: Unlikely" in predicted_answer:
                            predicted_answer = "Unlikely"
                        else:
                            predicted_answer = "Unknown"

                        # Record batch results
                        batch_results.append({
                            "id": id,
                            "predicted_answer": predicted_answer,
                            "ground_truth": label
                        })
                    except Exception as e:
                        print(f"Error: {e}, ID: {id}")

                # Add batch results to the overall results
                results.extend(batch_results)

                # Save results incrementally after every batch
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)

                pbar.update(1)
    return results

# Main function
def main(dataset_name, split, tokenizer, batch_size, save_path, device):
    dataset = load_hf_dataset(dataset_name, split)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    LLAMA_MODEL_HF_ID = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the actual model ID

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # BitsAndBytesConfig for int-4 optimization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_HF_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.eval()

# Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_HF_ID)

    # Assign eos_token as pad_token if no pad_token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Evaluate and save results
    results = evaluate(model, dataloader, tokenizer, device, save_path)

    # Save final results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

# Entry point
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_name", type=str, 
                        default="maximuspowers/nmb-plus-cleaned", help="Dataset name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="llama3.2-nlp.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args.dataset_name, args.split, None, args.batch_size, args.save_path, device)
