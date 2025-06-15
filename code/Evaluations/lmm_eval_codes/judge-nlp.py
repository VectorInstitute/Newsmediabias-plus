import pandas as pd
import openai
from tqdm import tqdm
import time
from openai import OpenAI
import os
import re
import argparse
import backoff

# Set your OpenAI API key here
openai.api_key = ''  # Replace with your actual OpenAI API key

# Initialize the OpenAI client
client = OpenAI(api_key=openai.api_key)

def create_evaluation_prompt(text, predicted_label):
    """
    Creates a prompt for the OpenAI API to evaluate if the given text is likely disinformation based on the predicted label.
    """
    disinformation_definition = (
        "Disinformation can be identified by the use of certain rhetorical techniques that aim to mislead or manipulate. Below is a checklist of common techniques used in disinformation:\n\n"
        "- Emotional Appeal: Uses language or imagery to provoke strong emotions (e.g., fear, anger) to distract from factual accuracy.\n"
        "- Exaggeration and Hyperbole: Makes unsupported claims or presents ordinary situations as extraordinary to manipulate perceptions.\n"
        "- Bias and Subjectivity: Unfairly favors one perspective while omitting key facts that would provide balance.\n"
        "- Repetition: Repeats misleading statements or biased points to embed them in the audience's mind.\n"
        "- Manipulative Word Choices: Uses emotionally charged or misleading terms to subtly influence opinions.\n"
        "- Appeals to Unqualified Authority: Cites sources or authorities lacking relevant expertise in the context.\n"
        "- Lack of Verifiable Sources: Relies on unverifiable or fabricated sources to support claims.\n"
        "- Logical Fallacies: Engages in flawed reasoning (e.g., circular logic, personal attacks) that undermines logical debate.\n"
        "- Conspiracy Theories: Promotes unproven, implausible theories often rooted in paranoia.\n"
        "- Inconsistencies and Factual Errors: Contains contradictions or easily disprovable inaccuracies, showing disregard for truth.\n"
        "- Selective Omission: Deliberately leaves out crucial information that would provide a more balanced understanding.\n"
        "- Manipulative Framing: Presents issues in a way that excludes alternative perspectives, focusing only on aspects that support a biased narrative.\n\n"
        "Evaluate the following text for the presence of these techniques and determine whether it is likely disinformation based on the predicted label."
    )
    prompt = (
        f"Evaluate if the following text is likely to be disinformation based on the predicted label.\n\n"
        f"Definition of Disinformation:\n{disinformation_definition}\n\n"
        f"Input Text: {text}\n\n"
        f"Predicted Label: {'Disinformation' if predicted_label == 1 else 'Not Disinformation'}\n\n"
        f"Unless there are strong reasons not to agree with this prediction, please confirm agreement. Do you agree with this prediction? Please respond with 'Agree' or 'Not Agree' and provide a very brief explanation for your decision."
    )
    return prompt

def get_api_response(prompt):
    """
    Sends a prompt to the OpenAI API and returns the response.
    """
    return client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", 
             "content": "You are an assistant that evaluates if a text is likely to be disinformation based on a given definition and predicted label."},
            {"role": "user", 
             "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.2,
    )

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Evaluate text for disinformation using OpenAI's GPT model.")
    parser.add_argument('--data_file', 
                        type=str, default='multi-modal-LLM/llama0-nlp-results.csv',
                        help='Path to the CSV file containing text and predicted labels for evaluation')
    parser.add_argument('--output_file', 
                        type=str, default='judge-llama-nlp-results.csv',
                        help='Path to save the evaluation results')
    
    args = parser.parse_args()


    # data_file = 'multi-modal-LLM/llama-ift-nlp-results.csv'

    data_file = args.data_file
    df = pd.read_csv(data_file)

    evaluation_results = []
    rank_scores = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Rows"):
        text = row['text']
        predicted_label = row['merged_predicted_label']
        
        prompt = create_evaluation_prompt(text, predicted_label)
        
        try:
            response = get_api_response(prompt)
            
            evaluation = response.choices[0].message.content.strip()
            
            # Print full response for debugging purposes
            print(f"GPT Response for row {index}: {evaluation}")
            
            # Check for 'Not Agree' first to avoid partial matches with 'Agree'
            if evaluation.startswith('Not Agree'):
                rank = 'Not Agree'
                # Capture everything after 'Not Agree'
                explanation = evaluation.split('Not Agree', 1)[1].strip()
            
            elif evaluation.startswith('Agree'):
                rank = 'Agree'
                # Capture everything after 'Agree'
                explanation = evaluation.split('Agree', 1)[1].strip()
            
            else:
                rank = None
                explanation = f"Unexpected response format: {evaluation}"
            
            # If no explanation is found after splitting, provide a default message
            if not explanation:
                explanation = "No detailed explanation provided."
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            rank = None
            explanation = f"Error during evaluation: {str(e)}"
        
        evaluation_results.append(explanation if explanation else "No explanation provided")
        rank_scores.append(rank if rank is not None else 'N/A')
        
        time.sleep(1)  # Respect API rate limits

    df['evaluation'] = evaluation_results
    df['rank'] = rank_scores

    output_file = args.output_file
    df.to_csv(output_file, index=False)

    print(f"Evaluation complete. Results saved to '{output_file}'.")

    # Count occurrences of 'Agree' and 'Not Agree' in the 'rank' column
    agree_counts = df['rank'].value_counts()

    # Print the counts of 'Agree' and 'Not Agree'
    print("Counts of 'Agree' and 'Not Agree':")
    print(agree_counts)

    # Calculate total number of evaluations
    total_evaluations = len(df)

    # Calculate percentages for 'Agree' and 'Not Agree'
    agree_percentage = (agree_counts.get('Agree', 0) / total_evaluations) * 100
    not_agree_percentage = (agree_counts.get('Not Agree', 0) / total_evaluations) * 100

    # Print the percentages
    print(f"\nPercentage of 'Agree': {agree_percentage:.2f}%")
    print(f"Percentage of 'Not Agree': {not_agree_percentage:.2f}%")