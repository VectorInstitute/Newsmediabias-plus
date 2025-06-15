import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
import time
import random
import wandb
import argparse
from codecarbon import EmissionsTracker
import logging
import re

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "llama3.1_classifier"

tracker = EmissionsTracker(project_name=f"{model_name}", 
                           output_dir=f"./{model_name}", output_file=f"{model_name}_emissions.csv")


class LlamaClassifier:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_length=500, shot_type="0-shot"):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.max_length = max_length
        self.shot_type = shot_type  # Determine whether 0-shot or 2-shot
        end_time = time.time() - start_time
        print("Time Taken for Initializing Model: ", end_time)

    def label_and_reason(self, content):
        if self.shot_type == "0-shot":
            full_prompt = f"""
            <|begin_of_text|>[INST]
            Assess the text below for potential disinformation.
            If you find any of the bias or disinformation, then the article is likely to be bias or disinformation; if the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
            Provide whether the text is 'Likely' or 'Unlikely' to be biased or disinformative without any further explanation.
            
            Response format required: 
            [Likely/Unlikely]

            Text: {content}
            [/INST]<|end_of_text|>
            """
        else:  # 2-shot mode
            full_prompt = f"""
            <|begin_of_text|>[INST]
            Assess the text below for potential disinformation (try finding deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.
            If you find any of the listed rhetorical techniques, then the article is likely disinformation; if not, it is likely not disinformation.
            Provide whether the text is 'Likely' or 'Unlikely' to be disinformation or biased for each without any further explanation.

            Examples:
            Text: Jenna Ellis, who assisted Donald Trump after the 2020 election then pleaded guilty last year in the Georgia election subversion case, has had her law license suspended in Colorado. The suspension begins July 2, according to a signed order from a state judge in Colorado. Ellis has been an attorney licensed to practice law in Colorado for more than a decade, according to court records. Ellis will be unable to practice law for three years in the state.
            Response: Likely

            Text: Canada’s two major freight railroads have shut their operations, according to management of the two companies, locking out 9,000 members of the Teamsters union who operate the trains and dealing a potential blow to both the Canadian and US economies.
            Response: Unlikely

            Text: {content}
            [/INST]<|end_of_text|>
            """
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, temperature=0.2, max_new_tokens=self.max_length)  
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
            
    def eval(self, text, true_vals):
        """Test the trained model"""

        self.model.eval()
        preds, targets = [], []
        tracker.start()
        for i, content in enumerate(text):
            response = self.label_and_reason(content)
            if response:
                analysis_str = response.replace('\n', ' ').replace('\r', ' ').strip()
                normalized_str = analysis_str.lower()

                pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
                label = re.search(pattern, normalized_str)
                if label:
                    label = label.group().title()
                print(i, label)
                if label == 'Likely' or label == 'Unlikely':
                    preds.append(label)
                    targets.append(true_vals[i])
        tracker.stop()
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, support = precision_recall_fscore_support(targets, preds, average='weighted')
        cm = confusion_matrix(targets, preds)
        report = classification_report(targets, preds, target_names=['Likely', 'Unlikely'], output_dict=True)

        print("\nClassification Results:")
        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1: {f1:.4f}")


        return accuracy, precision, recall, f1, support
        
    def predict(self, text):
        """Predict a label after training the model"""
        print("predicting the label for given input")
        self.model.eval()
        preds = []
        for content in text:
            response = self.label_and_reason(content)
            if response:
                analysis_str = response.replace('\n', ' ').replace('\r', ' ').strip()
                normalized_str = analysis_str.lower()

                pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
                label = re.search(pattern, normalized_str).group()
                if label:
                    label = label.title()
                preds.append(label)
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.1 Classifier")
    parser.add_argument("--test_file", type=str, default="Benchmarking/sample_test.csv", help="Path to the test data file")
    parser.add_argument("--shot_type", type=str, choices=["0-shot", "2-shot"], default="0-shot", help="Shot type for classification")
    args = parser.parse_args()



    test_data = pd.read_csv(args.test_file)
    
    test_inputs = test_data['content'].tolist()
    test_targets = test_data['text_label'].tolist()

    # Choose between 0-shot and 2-shot
    classifier = LlamaClassifier(max_length=1024, 
                                 shot_type=args.shot_type)  # Change to "2-shot" to use the 2-shot mode

    print("\nTest Set Evaluation:")
    classifier.eval(test_inputs, test_targets)

    test_sample = test_inputs[:5]
    predicted_labels = classifier.predict(test_sample)
    results_df = pd.DataFrame({
        'Text': test_sample,
        'Predicted Label': predicted_labels,
        'True Label': [label for label in test_targets[:5]]
    })
    print("\nSample Predictions:")
    print(results_df)
