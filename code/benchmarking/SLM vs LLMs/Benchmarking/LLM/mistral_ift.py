import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, MistralForCausalLM, AdamW, BitsAndBytesConfig
from peft import LoraConfig, get_peft_config, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import time
from codecarbon import EmissionsTracker
import logging
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import argparse
import random
import wandb
import re

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "mistral_ift"

tracker = EmissionsTracker(project_name=f"{model_name}", 
                           output_dir=f"./{model_name}", 
                           output_file=f"{model_name}_emissions.csv")


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weights)
        return ce_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
class MistralClassifier:
    def __init__(self, cache_dir, 
                 model_name="mistralai/Mistral-7B-Instruct-v0.3", 
                 num_labels=2, max_length=500):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = MistralForCausalLM.from_pretrained(model_name, 
                                                        cache_dir=cache_dir, 
                                                        quantization_config=bnb_config, 
                                                        num_labels=num_labels)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        peft_config = LoraConfig(task_type="CAUSAL_LM", 
                                 inference_mode=False, r=8, 
                                 lora_alpha=16, lora_dropout=0.1)
        self.model = get_peft_model(self.model, peft_config)

        self.model.to(self.device)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.scaler = GradScaler()
        end_time = time.time() - start_time
        print("Time Taken for Initializing Model: ", end_time)

    def prepare_data(self, text, labels, type, synthetic_balance=False, fit=False):
        """Prepare the data for training"""

        if os.path.exists(f'../{model_name}_{type}_dataset.pt') and os.path.exists(f'../{model_name}_{type}_dataset.pt'):
            dataset = torch.load(f'../{model_name}_{type}_dataset.pt')
            labels = torch.load(f'../{model_name}_{type}_labels.pt')
            return dataset, labels


        print(f"Preparing {type} data...")

        # Add Synthetic Data to balance classes
        if synthetic_balance and type=="Train":
            pass
        inputs = []
        for content in text:
            full_prompt = f"""
            [INST]
            Assess the text below for potential disinformation.
            If you find any of the bias or disinformation, then the article is likely to be bias or disinformation; if the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
            Provide whether the text is 'Likely' or 'Unlikely' to be biased or disinformative without any further explanation.
            
            Response format required: 
            [Likely/Unlikely]

            Text: {content}
            [/INST]
            """
            inputs.append(full_prompt)

        encodings = self.tokenizer(inputs, truncation=True, 
                                   padding="max_length", max_length=self.max_length)
        label_encodings = self.tokenizer(labels, truncation=True, 
                                         padding="max_length", max_length=self.max_length)

        input_ids = torch.tensor(encodings['input_ids'])
        label_ids = torch.tensor(label_encodings['input_ids'])

        assert input_ids.shape == label_ids.shape, f"Shape mismatch: {input_ids.shape} vs {label_ids.shape}"

        dataset = TensorDataset(input_ids, label_ids)

        torch.save(dataset, f'../{model_name}_{type}_dataset.pt')
        torch.save(labels, f'../{model_name}_{type}_labels.pt')

        return dataset, labels
        
    def train(self, train_dataset, train_encoded_labels, batch_size=100, 
              epochs=15, lr=2e-5, patience=5, k_folds=5):
        """Train on the train dataset with 5-fold cross-validation"""
        print("starting 5-fold cross-validation training...")


        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_results = []
        model_dir = f"Benchmarking/LLM/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        wandb.init(project="Benchmarking", name=model_name, config={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "k_folds": k_folds
        })
        tracker.start()
        for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
            print(f"\nTraining fold {fold + 1}...")
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(train_dataset, val_indices)
            
            train_dataloader = DataLoader(train_subset, sampler=RandomSampler(train_subset), batch_size=batch_size)
            val_dataloader = DataLoader(val_subset, sampler=SequentialSampler(val_subset), batch_size=batch_size)
            num_training_steps = epochs * len(train_dataloader)
            progress_bar = tqdm(range(num_training_steps))

            best_val_loss = float('inf')
            epochs_no_improve = 0
            train_losses, val_losses = [], []
            accumulation_steps = 4

            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0

                for i, batch in enumerate(train_dataloader):
                    batch = [r.to(self.device) for r in batch]
                    b_input_ids, b_labels = batch

                    # with autocast():
                    outputs = self.model(b_input_ids, labels=b_labels)
                    loss = outputs.loss
                    loss = loss / accumulation_steps
                    loss.backward()

                    # self.scaler.scale(loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        # self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        # self.scaler.step(optimizer)
                        # self.scaler.update()
                        optimizer.step()
                        optimizer.zero_grad()
                    total_loss += loss.item() * accumulation_steps
                    progress_bar.update(1)


                avg_train_loss = total_loss / len(train_dataloader)
                train_losses.append(avg_train_loss)
                logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{epochs} - Training loss: {avg_train_loss:.4f}')
                wandb.log({"Training Loss": avg_train_loss})

                self.model.eval()
                total_val_loss = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = [t.to(self.device) for t in batch]
                        b_input_ids, b_labels = batch
                        outputs = self.model(b_input_ids, labels=b_labels)
                        loss = outputs.loss
                        total_val_loss += loss.item()
                    
                    avg_val_loss = total_val_loss / len(val_dataloader)
                    val_losses.append(avg_val_loss)
                    logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{epochs} - Validation loss: {avg_val_loss:.4f}')
                    wandb.log({"Validation Loss": avg_val_loss})

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        self.model.save_pretrained(f'./{model_name}')
                        self.tokenizer.save_pretrained(f'./{model_name}')
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            logger.info('Early Stopping!')
                            break

            fold_results.append({'train_loss': train_losses, 'val_loss': val_losses})

        print("5-fold cross-validation complete.")
        tracker.stop()
        
        plt.figure(figsize=(10, 6))
        for fold, result in enumerate(fold_results):
            plt.plot(result['train_loss'], label=f'Fold {fold + 1} Training Loss')
            plt.plot(result['val_loss'], label=f'Fold {fold + 1} Validation Loss')
        plt.title('Training and Validation Loss Across 5 Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss') 
        plt.legend()
        plt.savefig(f'Benchmarking/LLM/{model_name}/loss_plot_5fold.png')
        plt.close()

    def eval(self, text, true_vals, batch_size=100):
        """Test the trained model"""
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for content in text:
                full_prompt = f"""Assess the text below for potential disinformation.
                    If you find any of the bias or disinformation, then the article is likely to be bias or disinformation; if the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
                    Provide whether the text is 'Likely' or 'Unlikely' to be biased or disinformative without any further explanation.
                    
                    Response format required: 
                    [Likely/Unlikely]

                    Text: {content}"""
                inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)

                output = self.model.generate(**inputs, temperature=0.2, max_new_tokens=self.max_length)
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                analysis_str = generated_text.replace('\n', ' ').replace('\r', ' ').strip()
                normalized_str = analysis_str.lower()

                pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
                label = re.search(pattern, normalized_str)
                if label:
                    label = label.group().title()
                if label == 'Likely' or label == 'Unlikely':
                    preds.append(label)
                    targets.append(true_vals[i])

        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, support = precision_recall_fscore_support(targets, preds, average='weighted')
        cm = confusion_matrix(targets, preds)
        report = classification_report(targets, preds, target_names=['Likely', 'Unlikely'], output_dict=True)

        print("\nClassification Results:")
        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")

        wandb.log({
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "support": support
        })

        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=self.label_encoder.classes_, columns=self.label_encoder.classes_))

        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=self.label_encoder.classes_, columns=self.label_encoder.classes_))
        print("\nClass-wise Metrics:")
        class_metrics = pd.DataFrame(report).transpose()
        class_metrics = class_metrics.drop(['accuracy', 'macro avg', 'weighted avg'])
        print(class_metrics.round(4))
        print("\nOverall Metrics:")
        print(pd.DataFrame(report['weighted avg'], index=['Weighted Avg']).round(4))

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'Benchmarking/LLM/{model_name}/confusion_matrix.png')
        plt.close()

        return accuracy, precision, recall, f1, support, cm, report
    
    def truncate_content(self, content, max_words=125):
        words = content.split()
        if len(words) > max_words:
            truncated_content = ' '.join(words[:max_words])
            return truncated_content
        return content

    
    def predict(self, text):
        """Predict a label after training the model"""
        print("predicting the label for given input")
        predicted_labels = []
        self.model.eval()
        with torch.no_grad():
            for content in text:
                full_prompt = f"""
                [INST]
                Assess the text below for potential disinformation.
                If you find any of the bias or disinformation, then the article is likely to be bias or disinformation; if the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
                Provide whether the text is 'Likely' or 'Unlikely' to be biased or disinformative without any further explanation.
                
                Response format required: 
                [Likely/Unlikely]

                Text: {self.truncate_content(content)}
                [/INST]
                """
                inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)

                output = self.model.generate(**inputs, temperature=0.2, max_new_tokens=self.max_length)
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                analysis_str = generated_text.replace('\n', ' ').replace('\r', ' ').strip()
                normalized_str = analysis_str.lower()

                pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
                label = re.search(pattern, normalized_str)
                if label:
                    label = label.group().title()

                predicted_labels.append(label)
                print(generated_text)
        return predicted_labels
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral Classifier Training and Evaluation")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache model and tokenizer")
    parser.add_argument("--train_file", type=str, default="Benchmarking/sample_train.csv", help="Path to the training data CSV file")
    parser.add_argument("--test_file", type=str, default="Benchmarking/sample_test.csv", help="Path to the test data CSV file")
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_file)
    test_data = pd.read_csv(args.test_file)
    
    train_inputs = train_data['content'].tolist()
    train_targets = train_data['text_label'].tolist()

    test_inputs = test_data['content'].tolist()
    test_targets = test_data['text_label'].tolist()

    classifier = MistralClassifier(num_labels=2, max_length=1024, cache_dir=args.cache_dir)

    train_dataset, train_encoded_labels = classifier.prepare_data(train_inputs, 
                                                                  train_targets, 
                                                                  type='Train', synthetic_balance=True, fit=True)

    print("\nTraining the Data on train_dataset...")
    classifier.train(train_dataset, train_encoded_labels, batch_size=4)
    print("\nTest Set Evaluation:")
    classifier.eval(test_inputs, test_targets)

    test_sample = test_inputs[:5]
    predicted_labels = classifier.predict(test_sample)
    print("\nSample Predictions:")
    for i, label in enumerate(predicted_labels):
        print('Text: ', test_sample[i])
        print('Predicted Label: ', predicted_labels[i])
        print('True Label: ', label)