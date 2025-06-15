import os
from transformers import MarianMTModel, MarianTokenizer
import random
import pandas as pd
import argparse
from nltk.corpus import wordnet
import nltk

nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)

model_en_to_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer_en_to_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

model_fr_to_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
tokenizer_fr_to_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')

def backtranslate(text):
    """
    Backtranslate the input text from English to French and then back to English.
    Args:
        text (str): The input text in English to be backtranslated.
    Returns:
        str: The backtranslated text in English.
    """
    translated = tokenizer_en_to_fr(text, 
                                    return_tensors="pt", 
                                    truncation=True, 
                                    padding=True, 
                                    max_length=500)
    translated_output = model_en_to_fr.generate(**translated)
    translated_text = tokenizer_en_to_fr.decode(translated_output[0], 
                                                skip_special_tokens=True)
    
    back_translated = tokenizer_fr_to_en(translated_text, 
                                         return_tensors="pt", 
                                         truncation=True, 
                                         padding=True, 
                                         max_length=500)
    back_translated_output = model_fr_to_en.generate(**back_translated)
    back_translated_text = tokenizer_fr_to_en.decode(back_translated_output[0], 
                                                     skip_special_tokens=True)
    return back_translated_text


def synonym_replacement(text, n=1):
    """
    Replace n words in the text with their synonyms.
    Args:
        text (str): The input text in which to replace words.
        n (int): The number of words to replace with synonyms.
    Returns:
        str: The text with n words replaced by their synonyms.
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if len(synonyms) >= 1:
            synonym = random.choice(list(set(synonyms)))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)


def load_checkpoint(checkpoint_file):
    """
    Load the checkpoint file to get the set of processed unique IDs.
    Args:
        checkpoint_file (str): The path to the checkpoint file.
    Returns:
        set: A set of unique IDs that have already been processed.
    """
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df['unique_id'].tolist())
    else:
        print("checkpoint file doesn't exist")
        processed_ids = set()  
    return processed_ids


def save_checkpoint(batch_df, checkpoint_file):
    if not os.path.exists(checkpoint_file):
        batch_df.to_csv(checkpoint_file, mode='w', 
                        header=True, index=False)
    else:
        batch_df.to_csv(checkpoint_file, mode='a', 
                        header=False, index=False)


if __name__ == "__main__":
    """
    Main function to perform backtranslation on the training data.
    It loads the training data, checks for already processed IDs, and applies backtranslation.
    """
    parser = argparse.ArgumentParser(description="Backtranslate training data.")
    parser.add_argument("--checkpoint_file", type=str, 
                        default='Benchmarking/backtranslated.csv', 
                        help="Path to the checkpoint file.")
    parser.add_argument("--train_data_file", 
                        type=str, 
                        default='Benchmarking/final-train_merged_file.csv', 
                        help="Path to the training data file.")
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data_file)
    checkpoint_file = args.checkpoint_file
    processed_ids = load_checkpoint(checkpoint_file)
    processed_ids = {str(id) for id in processed_ids}
    print(processed_ids)
    print(f"Initial size of train_data: {len(train_data)}")
    
    if processed_ids:
        train_data = train_data[~train_data['unique_id'].isin(processed_ids)]
    
    print(f"Size of train_data after filtering processed IDs: {len(train_data)}")

    train_unique_ids = train_data['unique_id'].tolist()
    train_inputs = train_data['first_paragraph'].tolist()
    train_targets = train_data['majority_final_label'].tolist()


    batch_size = 100
    backtranslated_data = []

    for i, (unique_id, text, target) in enumerate(zip(train_unique_ids, 
                                                      train_inputs, 
                                                      train_targets)):
        backtranslated_text = backtranslate(text)
        backtranslated_data.append({'unique_id': unique_id, 
                                    'first_paragraph': backtranslated_text, 
                                    'majority_final_label': target})

        if len(backtranslated_data) >= batch_size:
            batch_df = pd.DataFrame(backtranslated_data)
            save_checkpoint(batch_df, checkpoint_file)
            backtranslated_data.clear()

            print(f"Processed {i + 1} rows. Checkpoint saved.")

    if backtranslated_data:
        batch_df = pd.DataFrame(backtranslated_data)
        save_checkpoint(batch_df, checkpoint_file)
        print(f"Final checkpoint saved with {len(backtranslated_data)} rows.")
        backtranslated_data.clear()