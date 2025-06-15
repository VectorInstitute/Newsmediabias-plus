"""
Political idiology is extract from each news-article
"""

import pandas as pd
import re
import torch
import time
import asyncio
import argparse
from demo_utils import prompt, extract_info
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.asyncio import tqdm as tqdm_asyncio

async def predictions(model, tokenizer, messages):
    """
    Generate predictions using the model based on the provided messages.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.3,
            top_k=3
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, 
                            skip_special_tokens=True)

async def process_row(model, tokenizer, row, instruction, demo_type):
    """
    Process a single row of data to extract demographics using the model.
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"News article: {row['text_content']}"}
    ]

    response = await predictions(model, tokenizer, messages)
    #print(response)
    demographics = extract_info(response, demo_type)
    #print(demographics)
    
    return {
        "unique_id": row['unique_id'],
        f"{demo_type.lower()}": demographics
    }

async def main(args):
    start_time = time.time()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_CONFIG = {
        "name": args.model_name,
        "id": args.model_id,
    }

    data_path = args.data_path
    data_df = pd.read_csv(data_path) #.iloc[:20000,]

    model_name = args.model_name
    demo_type = args.demo_type.upper()

    instruction = prompt(demo_type)
    #print(instruction)

    cache_dir = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["id"], cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["id"],
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tasks = [process_row(model, tokenizer, row, instruction, demo_type) 
             for _, row in data_df.iterrows()]

    results = await tqdm_asyncio.gather(*tasks)

    df = pd.DataFrame(results)
    df.to_csv(f"{model_name}_{demo_type}demo.csv", index=False)
    print(df.info())

    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract demographics from news articles.")
    parser.add_argument("--model_name", type=str, default="llama3", help="Name of the model to use.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="ID of the model to use.")
    parser.add_argument("--data_path", type=str, default="./consolidated_data/cleaned_data.csv", help="Path to the data file.")
    parser.add_argument("--demo_type", type=str, default="IDIOLOGY", 
                        choices=["IDIOLOGY","GENDER", "RACE", "TARGETED_GROUP", "RELIGION"], 
                        help="Type of demographic to extract.")
    parser.add_argument("--cache_dir", type=str, default=".", help="Directory to cache the model.")
    args = parser.parse_args()
    asyncio.run(main(args))
