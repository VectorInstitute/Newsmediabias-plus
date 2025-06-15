import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Tokenizer,GPT2ForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import argparse
import os
import time
from codecarbon import EmissionsTracker
import logging
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import random
import wandb

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "gpt2"

tracker = EmissionsTracker(project_name=f"{model_name}", 
                           output_dir=f"./{model_name}", 
                           output_file=f"{model_name}_emissions.csv")


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, 
                                                    weight=self.weights)
        return ce_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, 
                                                    reduction="none", weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
class GPT2Classifier:
    def __init__(self, model_name="openai-community/gpt2", 
                 num_labels=2, max_length=500,
                 cache_dir = './cache'):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = GPT2ForSequenceClassification.from_pretrained(model_name, 
                                                                   cache_dir=cache_dir,
                                                                   num_labels=num_labels)
        self.model.config.pad_token_id = self.model.config.eos_token_id


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

        if fit:
            labels = self.label_encoder.fit_transform(labels)
        else:
            labels = self.label_encoder.transform(labels)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_instances = sum(counts)  
        for label, count in zip(unique_labels, counts):
            print(f'Class {self.label_encoder.inverse_transform([label])[0]}: {count} instances')
        print(f'Total instances: {total_instances}')  


        encodings = self.tokenizer(text, truncation=True, 
                                   padding=True, 
                                   max_length=self.max_length)
        print("labels are: ", labels)
        dataset = TensorDataset(
            torch.tensor(encodings['input_ids']),
            torch.tensor(encodings['attention_mask']), 
            torch.tensor(labels, dtype=torch.long)
        )

        torch.save(dataset, f'../{model_name}_{type}_dataset.pt')
        torch.save(labels, f'../{model_name}_{type}_labels.pt')

        return dataset, labels
    
    def calculate_class_weights(self, encoded_labels):
        """Encode the class labels to integers and calculate the class weights"""
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(encoded_labels), 
                                             y=encoded_labels)
        weights = torch.tensor(class_weights, dtype=torch.float)
        return weights.to(self.device)
    
    def train(self, train_dataset, train_encoded_labels, batch_size=100, 
              epochs=15, lr=2e-5, patience=5, k_folds=5):
        """Train on the train dataset with 5-fold cross-validation"""
        print("starting 5-fold cross-validation training...")


        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        fold_results = []
        model_dir = f"./{model_name}"
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
            
            train_dataloader = DataLoader(train_subset, 
                                          sampler=RandomSampler(train_subset), 
                                          batch_size=batch_size)
            val_dataloader = DataLoader(val_subset, 
                                        sampler=SequentialSampler(val_subset), 
                                        batch_size=batch_size)
            num_training_steps = epochs * len(train_dataloader)
            progress_bar = tqdm(range(num_training_steps))

            best_val_loss = float('inf')
            epochs_no_improve = 0
            train_losses, val_losses = [], []
            accumulation_steps = 4

            loss_fn = CrossEntropyLoss()
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0

                for i, batch in enumerate(train_dataloader):
                    batch = [r.to(self.device) for r in batch]
                    b_input_ids, b_input_mask, b_labels = batch

                    with autocast():
                        outputs = self.model(b_input_ids, token_type_ids=None, 
                                             attention_mask=b_input_mask)
                        loss = loss_fn(outputs.logits, b_labels)
                        loss = loss / accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                       max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
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
                        b_input_ids, b_input_mask, b_labels = batch
                        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                        loss = loss_fn(outputs.logits, b_labels)
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
        plt.savefig(f'./{model_name}/loss_plot_5fold.png')
        plt.close()

    def eval(self, dataset, batch_size=100):
        """Test the trained model"""
        test_dataloader = DataLoader(dataset, 
                                     sampler=SequentialSampler(dataset), 
                                     batch_size=batch_size)
        self.model.eval()

        predictions, true_vals = [], []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = [t.to(self.device) for t in batch]
                b_input_ids, b_input_mask, b_labels = batch
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                predictions.append(logits)
                true_vals.append(b_labels)

        predictions = torch.cat(predictions, dim=0)
        true_vals = torch.cat(true_vals, dim=0)
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        true_vals = true_vals.cpu().numpy()

        accuracy = accuracy_score(true_vals, preds)
        precision, recall, f1, support = precision_recall_fscore_support(true_vals, 
                                                                         preds, 
                                                                         average='weighted')
        cm = confusion_matrix(true_vals, preds)
        report = classification_report(true_vals, preds, 
                                       target_names=['Likely', 'Unlikely'], 
                                       output_dict=True)

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
        print(pd.DataFrame(cm, index=self.label_encoder.classes_, 
                           columns=self.label_encoder.classes_))

        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=self.label_encoder.classes_, 
                           columns=self.label_encoder.classes_))
        print("\nClass-wise Metrics:")
        class_metrics = pd.DataFrame(report).transpose()
        class_metrics = class_metrics.drop(['accuracy', 'macro avg', 'weighted avg'])
        print(class_metrics.round(4))
        print("\nOverall Metrics:")
        print(pd.DataFrame(report['weighted avg'], index=['Weighted Avg']).round(4))

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'./{model_name}/confusion_matrix.png')
        plt.close()

        return accuracy, precision, recall, f1, support, cm, report
    
    def predict(self, text):
        """Predict a label after training the model"""
        print("predicting the label for given input")
        self.model.eval()
        with torch.no_grad():
            encodings = self.tokenizer(text, truncation=True, 
                                       padding=True, max_length=self.max_length, 
                                       return_tensors='pt').to(self.device)
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            predicted_labels = [self.label_encoder.classes_[pred] for pred in predictions.cpu().numpy()]
        return predicted_labels
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GPT-2 model for text classification.")
    parser.add_argument("--cache_dir", type=str, 
                        default="./cache", help="Directory to cache model and tokenizer")
    parser.add_argument("--train_file", type=str, 
                        default="./Benchmarking/sample_train.csv", 
                        help="Path to the training data CSV file")
    parser.add_argument("--test_file", type=str, 
                        default="./Benchmarking/sample_test.csv", 
                        help="Path to the test data CSV file")
    args = parser.parse_args()



    train_data = pd.read_csv(args.train_file)
    test_data = pd.read_csv(args.test_file)
    
    train_inputs = train_data['content'].tolist()
    train_targets = train_data['text_label'].tolist()

    test_inputs = test_data['content'].tolist()
    test_targets = test_data['text_label'].tolist()

    classifier = GPT2Classifier(num_labels=2, max_length=1024,
                                cache_dir=args.cache_dir)

    train_dataset, train_encoded_labels = classifier.prepare_data(train_inputs, 
                                                                  train_targets, 
                                                                  type='Train', 
                                                                  synthetic_balance=True, fit=True)
    test_dataset, test_encoded_labels = classifier.prepare_data(test_inputs, 
                                                                test_targets, 
                                                                type='Test')

    print("\nTraining the Data on train_dataset...")
    classifier.train(train_dataset, train_encoded_labels, batch_size=32)
    print("\nTest Set Evaluation:")
    classifier.eval(test_dataset, batch_size=32)


    test_sample = test_inputs[:5]
    predicted_labels = classifier.predict(test_sample)
    results_df = pd.DataFrame({
        'Text': test_sample,
        'Predicted Label': predicted_labels,
        'True Label': [classifier.label_encoder.classes_[label] for label in test_encoded_labels[:5]]
    })
    print("\nSample Predictions:")
    print(results_df)