import os
import inspect
from PIL import Image
import torch
import pandas as pd
import time
import re
import argparse
from transformers import AutoTokenizer, AutoModel

###########Helper Function##############

def cpm_classifier(img_folder,images, model,question, tokenizer):
    """
    Classifies images using the CPM model and returns a DataFrame with image IDs, labels, and reasons.
    Args:
        img_folder (str): Path to the folder containing images.
        images (list): List of image filenames to classify.
        model (transformers.AutoModel): The pre-trained CPM model for classification.
        question (str): The question to ask the model for classification.
        tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
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
        #Checking the output of model.chat method
            msgs = [{'role': 'user', 'content': question}]
            result = model.chat(image=image, 
                                msgs=msgs, 
                                context=None, 
                                tokenizer=tokenizer, 
                                sampling=True, 
                                temperature=0.7)
            if isinstance(result, tuple) and len(result) == 3:
                res, context, _ = result
            else:
                res = result  # If it returns a single value, assume it's the response
                context = None  # Or some default value
            
            lines = res.strip().split('\n')
            label = next((line.split(":", 1)[1].strip().lower() for line in lines if line.lower().startswith("label:")), 'unknown')
            reason = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("reason:")), 'No reasoning provided')
            
            if label not in ['biased', 'unbiased']:
                label = 'unknown'
                reason = 'Invalid label provided by the model'
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            label = 'unknown'
            reason = 'Error during classification'

        classes.append(label)
        reasons.append(reason)
    
    return pd.DataFrame({"img_id": img_ids, 
                         "cpm_labels": classes, 
                         "cpm_reasons": reasons})


# Batch processing function
def process_batch(img_folder,images, model, question,tokenizer, out_file,batch_size,checkpoint_file):
    """
    Processes images in batches, classifies them using the CPM model, and saves results to a CSV file.
    Args:
        img_folder (str): Path to the folder containing images.
        images (list): List of image filenames to classify.
        model (transformers.AutoModel): The pre-trained CPM model for classification.
        question (str): The question to ask the model for classification.
        tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        out_file (str): Path to the output CSV file.
        batch_size (int): Number of images to process in each batch.
        checkpoint_file (str): Path to the checkpoint file to resume processing.
    """
    start_idx = 0
    items_processed = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_idx = int(f.read().strip())

    while start_idx < len(images):
        end_idx = min(start_idx + batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        df=cpm_classifier(img_folder,batch_images, model,question, tokenizer)
  
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

########### run the model #############

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Run CPM image classification")
    parser.add_argument('--img_folder', type=str, default="consolidated_data/images", help="Path to the folder containing images")
    args = parser.parse_args()
    # Current and parent directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_folder = args.img_folder
    images=os.listdir(img_folder)[:15]


    question = """You are a news image classifier bot. The news image can be classified as:
        1. Unbiased (Representing unbiased image)
        2. Biased (represents wrong portrayal, hatred gesture, propaganda, and satire).

        Your task is to look at the following images, classify them using the above definitions, and provide the reason for your classification. 
        Answer should be in the format:
        Label: [biased/unbiased]
        Reason: [your reasoning here]"""
    
    # Model initialization
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', 
                                      trust_remote_code=True, 
                                      torch_dtype=torch.bfloat16)
    model = model.to(device='cuda', 
                     dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', 
                                              trust_remote_code=True)
    model.eval()

    # Number of samples in each batch
    batch_name='batch0_4k'
    out_file = os.path.join(current_dir, "outputs", f"{batch_name}_cpm.csv")
    start_time=time.time()
    checkpoint_file=f"{batch_name}_cpm.txt"
    batch_size=5
    process_batch(img_folder,
                  images, 
                  model, 
                  question,
                  tokenizer,
                  out_file,
                  batch_size,
                  checkpoint_file)

    end_time=time.time()
    hours=(end_time-start_time)//3600
    minutes=(end_time-start_time)%3600//60

    print(f"Total time taken: hours: {hours}, and minutes {minutes}")

    print("Successful")



