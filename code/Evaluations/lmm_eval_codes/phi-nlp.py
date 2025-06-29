import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import logging
import textwrap
import argparse

torch.cuda.empty_cache()
torch.manual_seed(0)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhiClassifier:
    def __init__(self, model_name="microsoft/Phi-3-mini-128k-instruct", max_length=1500):
        """Initialize the model class."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.max_length = max_length

    def label_and_reason(self, content):
        """Generate the model's raw response for the input content and extract the label."""
        full_prompt = textwrap.dedent(f"""\
        [INST]
        Assess the text below for potential disinformation.
        Provide whether the text is '[Likely]' or '[Unlikely]' to be biased or disinformative. Do not provide any explanation.
        Your answer must be either '[Likely]' or '[Unlikely]' based on your assessment.
        Text: {content}
        [/INST]
        """)
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    temperature=0.2,
                    max_new_tokens=self.max_length,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            print(response)
            label = "[No Match Found]"
            if response.endswith("[Likely]"):
                label = "[Likely]"
            elif response.endswith("[Unlikely]"):
                label = "[Unlikely]"
            return response, label
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "[Error]", "[Error]"

# Utility function for incremental processing
def save_checkpoint(file_path, unique_id):
    with open(file_path, "w") as f:
        f.write(str(unique_id))

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi NLP Evaluation Script")
    parser.add_argument("--input_path", type=str, default="dataset.csv", help="Path to the input dataset CSV file.")
    parser.add_argument("--output_path", type=str, default="Phi.csv", help="Path to save the output CSV file.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint-p.txt", help="Path to save the checkpoint file.")
    args = parser.parse_args()



    input_path = args.input_path
    output_path = args.output_path
    checkpoint_path = args.checkpoint_path

    # Load dataset
    dataset = pd.read_csv(input_path)
    required_columns = ['unique_id', 'content', 'text_label']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")
    dataset = dataset[required_columns]

    # Checkpointing
    last_processed_id = load_checkpoint(checkpoint_path)
    if last_processed_id:
        logger.info(f"Resuming from unique_id: {last_processed_id}")
        dataset = dataset[dataset['unique_id'] > last_processed_id]

    # Initialize model
    classifier = PhiClassifier(max_length=1024)

    # Incremental saving
    if not os.path.exists(output_path):
        dataset.head(0).to_csv(output_path, index=False)  # Save header only if file doesn't exist

    for idx, row in dataset.iterrows():
        try:
            content = row['content']
            response, label = classifier.label_and_reason(content)
            row['Model Response'] = response
            row['Extracted Label'] = label

            # Save the processed row
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)

            # Save checkpoint
            save_checkpoint(checkpoint_path, row['unique_id'])
            logger.info(f"Processed unique_id: {row['unique_id']}")
        except Exception as e:
            logger.error(f"Failed to process unique_id {row['unique_id']}: {e}")
