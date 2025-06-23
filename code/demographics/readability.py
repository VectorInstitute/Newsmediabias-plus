import pandas as pd
import re
import argparse

# This module provides functions to calculate the Flesch-Kincaid readability index for text data.

##########Count_syllables ############
def count_syllables(word):
    """Count the number of syllables in a word."""
    word = word.lower()
    syllable_count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith("e"):
        syllable_count -= 1
    if syllable_count == 0:
        syllable_count = 1
    return syllable_count

#####Readability_index calculator ###########
def readability_index(text):
    """Compute the Flesch-Kincaid readability index of a text."""
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    num_sentences = len(sentences) - 1  # To exclude the last empty split

    # Split text into words
    words = re.findall(r'\b\w+\b', text)
    num_words = len(words)
    
    # Calculate total syllables
    num_syllables = sum(count_syllables(word) for word in words)
    
    # Flesch-Kincaid grade level formula
    if num_sentences > 0 and num_words > 0:
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        return fk_grade
    else:
        return 0.0  # Return 0 if no sentences or words are present
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate readability index for text data.")
    parser.add_argument("--data_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_path", type=str, default="./readability.csv", help="Path to save the output CSV file.")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.data_path)

    # Calculate readability index for each row
    readabilities = [readability_index(row['text_content']) for _, row in df.iterrows()]
    df['readability'] = readabilities

    # Save the results to a new CSV file
    df.to_csv(args.output_path, index=False)