import torch
import os
import json
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import logging
import time

# === LOGGING ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === CONFIGURATION ===
MODEL_DIR = ""  # Directory to load the model from (if using local)
RESULTS_FILE = "results_Janus_Pro.json"  # File to save results
# IMAGE_FOLDER = "./images_1/" # If images are stored locally

# Resize image to manageable size
def load_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Load dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

# Load model and processor
def load_model_and_processor(model_path, source="local"):
    print(f"Loading Janus-Pro model from {'local directory' if 'local' in model_path else 'Hugging Face'}...")
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
    return model, processor, tokenizer

# Truncate content
def truncate_to_2000_words(text):
    words = text.split()
    return ' '.join(words[:2000]) if len(words) > 2000 else text

# Save results to file
def save_results(results, results_file):
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


# Process a single sample
def process_sample(sample_id, dataset_sample, processor, model, tokenizer, save_path):
    # If image is stored locally
    # img_path = os.path.join(IMAGE_FOLDER, f"{sample_id}.jpg")
    # image = load_image(img_path)
    image = sample["image"]
    if image is None:
        print(f"Skipping sample {sample_id} (image not found)")
        return None
    
    article_text = truncate_to_2000_words(dataset_sample["article_text"])
    answer_gt = dataset_sample["multimodal_label"]
    
    prompt = "Assess the text and image below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.\n" \
            "If you find any of the listed rhetorical  techniques, then the article is likely disinformation; if not, it is likely not disinformation.\n\n" \
            "Rhetorical Techniques Checklist:\n" \
            "- Emotional Appeal: Uses language that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.\n" \
            "- Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.\n" \
            "- Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.\n" \
            "- Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.\n" \
            "- Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.\n" \
            "- Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.\n" \
            "- Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.\n" \
            "- Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.\n" \
            "- Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.\n" \
            "- Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.\n" \
            "- Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.\n" \
            "- Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.\n\n" \
            f"Article: {article_text}\n\n" \
            "Please provide your answer in the format: 'Likely' or 'Unlikely' and do not provide any explaination or text." 

    
    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [image]},
        {"role": "<|Assistant|>", "content": ""}
    ]

    # Preprocess input
    prepared_inputs = processor(
        conversations=conversation, 
        images=[image], 
        force_batchify=True).to(model.device)
    
    # Generate response
    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=16,
            do_sample=False,
        )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return {"id": sample_id, 
            "ground_truth": answer_gt, 
            "predicted_answer": answer}



# === MAIN SCRIPT ===
if __name__ == "__main__":
    # Load dataset and existing results
    dataset_name = "vector-institute/newsmediabias-plus-clean"
    dataset_split = "train"
    dataset =  load_hf_dataset(dataset_name, dataset_split)
    # Load model and processor
    hf_model_path = "deepseek-ai/Janus-Pro-7B"
    model, processor, tokenizer = load_model_and_processor(hf_model_path, source="huggingface") # Load from Hugging Face
    # model, processor, tokenizer = load_model_and_processor(MODEL_DIR, source="local") # Load from local directory
    
    # Process samples
    results = []
    save_path = RESULTS_FILE
    for sample in tqdm(dataset):
        sample_id = sample["unique_id"]
        
        # Process the sample
        result = process_sample(sample_id, sample, processor, model, tokenizer, save_path)
        if result:
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