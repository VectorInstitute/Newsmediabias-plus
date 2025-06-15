import torch
import os
import json
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import logging
import time



# === CONFIGURATION ===
MODEL_DIR = ""  # Directory to load the model from (if using local)
RESULTS_FILE = "results_deepseek_llm.json"  # File to save results

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

# Truncate content
def truncate_to_2000_words(text):
    words = text.split()
    return ' '.join(words[:2000]) if len(words) > 2000 else text

# Save results to file
def save_results(results, results_file):
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

# Load model and processor
def load_model_and_processor(model_path, source="local"):
    print(f"Loading Deepseek model from {'local directory' if source == 'local' else 'Hugging Face'}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer

# Process a single sample
def process_sample(sample_id, dataset_sample, model, tokenizer):
    article_text = truncate_to_2000_words(dataset_sample["article_text"])
    answer_gt = dataset_sample["multimodal_label"]

    prompt = (
            "Please analyze the following article and determine whether it contains any signs of disinformation. "
            "If you find the content to be misleading, biased, or negative, respond with 'Classification: Likely.' "
            "If the content seems accurate, credible, and unbiased, respond with 'Classification: Unlikely.' "
            "Please choose only one classification.\n\n"
            f"Article: {article_text}\n\n"
        )

    messages = [
            {"role": "user", 
             "content": prompt}
        ]
    
    input_tensor = tokenizer.apply_chat_template(messages, 
                                                 add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_tensor.to(model.device), 
                                 max_new_tokens=16)

    # Decode the model output
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    return {"id": 
            sample_id, "ground_truth": 
            answer_gt, "predicted_answer": result}



# === MAIN SCRIPT ===
if __name__ == "__main__":
    # Load dataset and existing results
    dataset_name = "vector-institute/newsmediabias-plus-clean"
    dataset_split = "train"
    dataset = load_hf_dataset(dataset_name, dataset_split)
    print("Dataset size: ", len(dataset))
    
    # Load model and processor
    hf_model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    model, tokenizer = load_model_and_processor(hf_model_path, source="huggingface") # Load from Hugging Face
    # model, processor, tokenizer = load_model_and_processor(MODEL_DIR, source="local") # Load from local directory
    
    # Process samples
    results = []
    save_path = RESULTS_FILE
    for sample in tqdm(dataset):
        sample_id = sample["unique_id"]
        # Process the sample
        result = process_sample(sample_id, sample, model, tokenizer)
        results.append(result)
        if len(results) % 100 == 0:
                    intermediate_save_path = save_path.replace(".json", f"_intermediate_{len(results)}_{time.strftime('%Y%m%d-%H%M%S')}.json")
                    curr_path_name = intermediate_save_path
                    with open(intermediate_save_path, "w") as f:
                        json.dump(results, f, indent=4, default=str)
                    logger.info(f"Intermediate results saved to {intermediate_save_path}.")
                    if prev_path_name != "":
                        # Delete prev_content
                        logger.info(f"Deleting previous file named: {prev_path_name}")
                        os.remove(prev_path_name)
                        prev_path_name = curr_path_name
                    else:
                        prev_path_name = curr_path_name

    # Save final results
    save_results(results, RESULTS_FILE)