import torch
from torch.utils.data import DataLoader
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Features, Value, Image as HFImage
from argparse import ArgumentParser
import json
import time
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import os
from PIL import Image

# Define dataset features
features = Features({
    "unique_id": Value("string"),
    "outlet": Value("string"),
    "headline": Value("string"),
    "image_description": Value("string"),
    "article_text": Value("string"),
    "image": HFImage(),
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
        'images': [item['image'] for item in batch],
        'article_texts': [item['article_text'] for item in batch],
        'answers': [item['multimodal_label'] for item in batch],
        'ids': [item['unique_id'] for item in batch]
    }

def evaluate(model, dataloader, processor, device, save_path):
    model.eval()
    results = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            for batch in dataloader:
                images, article_texts, labels, ids = (
                    batch['images'], 
                    batch['article_texts'], 
                    batch['answers'], 
                    batch['ids']
                )
                batch_results = []  # Store results for the current batch
                for img, text, label, id in zip(images, article_texts, labels, ids):
                    try:
                        text = truncate_to_2000_words(text)
                        prompt = (
                            "Review the following article and the associated image for disinformation. "
                            "Respond with 'Classification: Likely' if the article or image seems misleading, "
                            "and 'Classification: Unlikely' if they seem accurate and credible. "
                            "Provide only one of the two classifications.\n\n"
                            f"Article: {text}\n\n"
                        )

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": str(prompt)},
                                ]
                            }
                        ]
                        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        inputs = processor(
                            img,  # PIL image
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(device)

                        # output = model.generate(**inputs, max_new_tokens=50)
                        output = model.generate(**inputs, 
                                                max_new_tokens=50, 
                                                temperature=0.7, 
                                                top_p=0.9)

                        predicted_answer = processor.decode(output[0], skip_special_tokens=True)

                        # Clean and extract the predicted answer
                        if "Classification: Likely" in predicted_answer:
                            predicted_answer = "Likely"
                        elif "Classification: Unlikely" in predicted_answer:
                            predicted_answer = "Unlikely"
                        else:
                            predicted_answer = "Unknown"

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
def main(dataset_name, split, processor, batch_size, save_path, device):
    dataset = load_hf_dataset(dataset_name, split)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    LLAMA_MODEL_HF_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(
        LLAMA_MODEL_HF_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(LLAMA_MODEL_HF_ID)
    # Update the main function
    results = evaluate(model, dataloader, processor, device, save_path)


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
    parser.add_argument("--save_path", type=str, default="llama3.2.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args.dataset_name, args.split, None, args.batch_size, args.save_path, device)
