import os
import inspect
from PIL import Image
import torch
import pandas as pd
import time
import re
import argparse
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,BitsAndBytesConfig

###########Helper Function##############

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

def llava_classifier(img_folder,images, model,prompt):
    """
    Classifies images using the LLaVA model and returns a DataFrame with image IDs, labels, and reasons.
    Args:
        img_folder (str): Path to the folder containing images.
        images (list): List of image filenames to classify.
        model (LlavaNextForConditionalGeneration): Pre-trained LLaVA model.
        prompt (str): Prompt to use for classification.
    Returns:
        pd.DataFrame: DataFrame containing image IDs, labels, and reasons.
    """
    classes, img_ids, reasons = [], [], []
    for img in images:
        img_path=img_folder+f"/{img}"

        try:
            image = Image.open(img_path).convert('RGB')
            img_ids.append(os.path.splitext(os.path.basename(img_path))[0])
            
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            img_ids.append(os.path.splitext(os.path.basename(img_path))[0])
            classes.append('unknown')
            reasons.append('Error opening image')
            continue
        
        try:
            processor=processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
            processor.tokenizer.padding_side = "left"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
            generate_ids = model.generate(**inputs, max_new_tokens=150)
            output=processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            lines = output.strip().split("[/INST] ", 1)[1]
            
            # Regular expression patterns
            label_pattern = r"Label:\s*(.+)"
            reason_pattern = r"Reason:\s*(.+)"

            # Extract label and reason using regex
            label_match = re.search(label_pattern, lines)
            reason_match = re.search(reason_pattern, lines)

            label = label_match.group(1) if label_match else None
            reason = reason_match.group(1) if reason_match else None
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            label = 'unknown'
            reason = 'Error during classification'

        classes.append(label)
        reasons.append(reason)
    
    return pd.DataFrame({"img_id":img_ids, 
                         "LLaVAlabels":classes, 
                         "LLVANextReason":reasons})


# Batch processing function
def process_batch(img_folder,images, model,prompt, out_file,batch_size,checkpoint_file):
    """
    Processes images in batches, classifies them using the LLaVA model, and saves results to a CSV file.
    """
    start_idx = 0
    items_processed = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_idx = int(f.read().strip())

    while start_idx < len(images):
        end_idx = min(start_idx + batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        df=llava_classifier(img_folder,batch_images, model,prompt)
  
        if start_idx == 0:
            df.to_csv(out_file, index=False)
            #print(f"Processed and saved {len(df)} items")
        else:
            df.to_csv(out_file, mode='a', header=False, index=False)
            #print(f"Processed and saved {len(df)} items")

        time.sleep(2)
        start_idx = end_idx
        items_processed += len(batch_images)
        with open(checkpoint_file, 'w') as f:
            f.write(str(start_idx))

        print(f"Processed and saved {items_processed} items")

    if start_idx >= len(images):
        os.remove(checkpoint_file)
        print("Processing complete. Checkpoint file removed.")

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="LLaVA Image Classifier")
    parser.add_argument('--img_folder', type=str, default="consolidated_data/images", help="Path to the folder containing images")
    args = parser.parse_args()
                        

    # Current and parent directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_folder = args.img_folder
    images=os.listdir(img_folder)[:15]


    prompt = """[INST] <image>\nYou are a news image classifier bot. The news image can be classified as:\
            1. Unbiased (Representing unbiased image)\
            2. Biased (represents wrong portrayal, hatred gesture, propaganda, and satire).\

            Your task is to look at the following images, classify them using the above definitions, and provide the reason for your classification.\ 
            Answer should be in the format:
            Label: [biased/unbiased]
            Reason: [your reasoning here] [/INST]"""
    
    # Quantization specifier
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                             bnb_4bit_quant_type="nf4", 
                                             bnb_4bit_compute_dtype=torch.float16,)
    model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME,
                                                              quantization_config=quantization_config, 
                                                              device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Number of samples in each batch
    batch_name='batch0_15'
    out_file = os.path.join(current_dir, "outputs", f"{batch_name}_llava.csv")
    start_time=time.time()
    checkpoint_file=f"{batch_name}_llava.txt"
    batch_size=5
    process_batch(img_folder,images, model,prompt, out_file,batch_size,checkpoint_file)

    end_time=time.time()
    hours=(end_time-start_time)//3600
    minutes=(end_time-start_time)%3600//60

    print(f"Total time taken: hours: {hours}, and minutes {minutes}")

    print("Successful")

