import transformers
import torch
import pandas as pd
import os
import csv
import argparse
from tqdm import tqdm

# This script uses the LLama3 model to apply linguistic transformations to text articles.

class LLama3:
    def __init__(self):
        """Initialize the LLama3 model for text generation."""
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def apply_mr(self, article, mr):
        """Apply the specified modification request (mr) to the article.
        Args:
            article (str): The original article to be modified.
            mr (str): The modification request type (e.g., "mr1", "mr2", "mr3").
        Returns:
            str: The modified article based on the specified modification request.
        """

        # mr1 is for active/passive voice transformation
        if mr == "mr1":
            prompt = f"""Please review the article below and transform all sentences by converting active voice to passive voice and vice versa, where appropriate.

            Provide only the revised article, starting with the term "new_article:"

            Original Article:
            {article}
            """
            
        # mr2 is for double negation
        if mr == "mr2":
            prompt = f"""Please review the article below and transform affirmative sentences into double negation sentences. (Double negation means using two negative elements within a clause or sentence, typically leading to a positive implication.) Ensure that the transformations maintain equivalent meanings.

            Provide only the revised article, starting with the term "new_article:"

            Article:
            {article}
            """
        
        # mr3 is for synonym replacement
        if mr == "mr3":
            prompt = f"""Review the article below and Replace key words with their synonyms. Ensure that the transformed sentences retain equivalent meanings.

            Provide only the revised article, without extra information, starting with the term "new_article:"

            Article:
            {article}
            """
        messages = [
            {"role": "system", 
                "content": "You are a helpful assistant."},
            {"role": "user", 
                "content": prompt},
        ]

        try:

            outputs = self.pipeline(
                messages,
                max_new_tokens=len(article)+20,
            )
            
            ans = outputs[0]["generated_text"][-1]['content'].split("\n", 1)[-1]
            if ans.startswith("\n"):
                ans = ans[1:]
            return ans        
        except Exception as e:
            
            print(f"Error querying model: {e}")
            return None


def get_create_output_df(output_file_name, mr):   
    """
    Create or load the output DataFrame for storing modified articles.
    Args:
        output_file_name (str): The name of the output CSV file.
        mr (str): The modification request type (e.g., "mr1", "mr2", "mr3").
    Returns:    
        pd.DataFrame: A DataFrame containing the modified articles.
    """ 
    if not os.path.exists(output_file_name):
        data = [['original_article', 
                 'article_mr1', 
                 'article_mr2', 
                 'article_mr3']]

        with open(output_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    return pd.read_csv(output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply linguistic transformations to text using LLaMA3.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file (must contain 'first_paragraph' column), e.g., ../Data/train.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--mr_type", type=str, required=True, choices=["mr1", "mr2", "mr3"], help="Modification request type")
    args = parser.parse_args()

    bot = LLama3()
    input_df = pd.read_csv(args.input_csv,
                        usecols=['first_paragraph'])
    mr = args.mr_type
    output_file_name = args.output_csv
    output_df = get_create_output_df(output_file_name, mr)
    last_index = len(output_df)

    print("contunuing saving from index...", last_index)

    buffer = [] 

    for index, row in tqdm(input_df.iterrows(), total=len(input_df)):
        if index < last_index:
            continue
        original_article = row['first_paragraph']
        new_article = bot.apply_mr(original_article, mr)
        
        if mr == "mr1":
            buffer.append([original_article, new_article, '', ''])
        elif mr == "mr2":
            buffer.append([original_article, '', new_article, ''])
        elif mr == "mr3":
            buffer.append([original_article, '', '', new_article])

        if len(buffer) == 10 or index == len(input_df) - 1:
            with open(output_file_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(buffer)
            
            buffer = []
            print("Saved up to index", index)
    print("All articles processed and saved.")

# To run the script, use the following command:
# python articleChanger.py --input_csv ../Data/train.csv --output_csv ../Data/modified_articles.csv --mr_type mr1