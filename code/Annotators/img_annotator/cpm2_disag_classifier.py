import os
import inspect
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import torch
import pandas as pd
import time
import numpy as np
import argparse
from utils import file_paths,extract_image_paths,_img_paths

MODEL_NAME = 'openbmb/MiniCPM-V-2'


# Classification function
def miniCPMClassifier(img_file_dir,df, model, tokenizer):
    """
    Classifies images as biased or unbiased based on the provided context and image content.
    Args:
        img_file_dir (str): Directory containing the images.
        df (pd.DataFrame): DataFrame containing image metadata.
        model: Pre-trained model for classification.
        tokenizer: Tokenizer for the model.
    Returns:
        pd.DataFrame: DataFrame containing image IDs, classification labels, and reasoning.
    """
    classes, img_ids, reasons = [], [], []
    for index, row in df.iterrows():
        img_path=img_file_dir+f"/{row['image']}"
        #print(img_path)
        title=row['title']
        date=row['date']

        context=f"Headline: {title}, Date: {date}"
        #print(context)

        prompt = f"""Analyze the given image and the following context, if present, from a news article and perform the following tasks.
    
        Context: {context}
        
        tasks:
        1.Consider the following, and any other relevant factors, in your analysis (mention the key factors you used in the analysis) and return your analysis as Reason:
        - Consider the headline and pulication date (if available) of the image and analyze how the image might influence the reader's interpretation of the headline.
        - A potential biased image can represent wrong portrayal, hatred gesture, propaganda, and satire.
        - An unbiased image depicts the subject or event without manipulating the viewer's perception through selective editing, misleading angles, or omission of context. It should provide a fair representation, avoiding stereotypes and ensuring diversity and inclusivity. Ethical standards must be upheld to maintain the integrity and truthfulness of the visual reporting

        2. Classify the image as biased or unbiased, and provide the reason for your classification. 
        
        Answer should be in the format:
        Label: [biased/unbiased]
        Reason: [your reasoning here]

        """
        question=prompt
        #print(question)
        msgs = [{'role': 'user', 'content': question}]
        try:
            image = Image.open(img_path).convert('RGB')
            img_ids.append(row["dis_agg_id"])
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            img_ids.append(row["dis_agg_id"])
            classes.append('unknown')
            reasons.append('Error opening image')
            continue
        
        try:
            result = model.chat(image=image, msgs=msgs, context=None, tokenizer=tokenizer, sampling=True, temperature=0.7)
            if isinstance(result, tuple) and len(result) == 3:
                res, context, _ = result
            else:
                res = result  # If it returns a single value, assume it's the response
                context = None  # Or some default value
            
            lines = res.strip().split('\n')
            label = next((line.split(":", 1)[1].strip().lower() for line in lines if line.lower().startswith("label:")), 'unknown')
            reasoning = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("reason:")), 'No reasoning provided')

            if label not in ['biased', 'unbiased']:
                label = 'unknown'
                reasoning = 'Invalid label provided by the model'
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            label = 'unknown'
            reasoning = 'Error during classification'
        
        classes.append(label)
        reasons.append(reasoning)
    return pd.DataFrame({"img_id": img_ids, 
                         "cpm_labels": classes, 
                         "cpm_reasons": reasons})


# Batch processing function
def process_batch(img_file_dir,data, model, tokenizer, out_file, batch_size, checkpoint_file):
    """
    Processes the data in batches, saving results to a CSV file and using a checkpoint file to resume processing if interrupted.
    """
    start_idx = 0
    items_processed = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_idx = int(f.read().strip())

    while start_idx < len(data):
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx,]
        df = miniCPMClassifier(img_file_dir,batch_data, model, tokenizer)

        if start_idx == 0:
            df.to_csv(out_file, index=False)
            #print(f"Processed and saved {len(df)} items")
        else:
            df.to_csv(out_file, mode='a', header=False, index=False)
            #print(f"Processed and saved {len(df)} items")

        #time.sleep(2)
        start_idx = end_idx
        items_processed += len(batch_data)
        with open(checkpoint_file, 'w') as f:
            f.write(str(start_idx))

        print(f"Processed and saved {items_processed} items")

    if start_idx >= len(data):
        os.remove(checkpoint_file)
        print("Processing complete. Checkpoint file removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with MiniCPM-V-2 model.")
    parser.add_argument('--data_dir', type=str, default='consolidated_data/', help='Directory containing the data files')

    args = parser.parse_args()

    # Get current and parent directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder_name='batch3k_3500'
    # Configuration
    out_file = os.path.join(current_dir, "outputs", f"{folder_name}_cpmv2.csv")
    img_file_dir = os.path.join(args.data_dir, "images")
    cont_file_dir = os.path.join(args.data_dir, "cleaned_data.csv")

    new_df=pd.read_csv("./Dataset/test_title.csv")

    df=new_df.iloc[3000:3500,]

    print(f"The total number of files in {folder_name} are {len(df)}")

    # Load model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME, 
                                      trust_remote_code=True, 
                                      torch_dtype=torch.bfloat16)
    model = model.to(device='cuda', 
                     dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                              trust_remote_code=True)
    model.eval()

    # Start running labelling
    start_time = time.time()
    checkpoint_file = f"{folder_name}_cpm.txt"
    batch_size = 200

    process_batch(img_file_dir,df, model, tokenizer, out_file, batch_size, checkpoint_file)

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes = rem // 60

    print(f"Total time taken: {(hours)} hours and {(minutes)} minutes")
    print("Successful")