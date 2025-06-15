import os
import pandas as pd
import argparse
from openai import OpenAI



def analyze_and_label_text(text, client):
    """
    Analyzes the given text for rhetorical techniques and disinformation using OpenAI's GPT-4o-mini model.
    """
    prompt = f"""
    Given the text below, identify the presence of the listed rhetorical techniques, if any of them is present, then mark its presence.
    
    Text: {text}
    
    Rhetorical Techniques:
    1. Emotional Appeal - Uses emotionally charged language to provoke strong reactions.
    2. Exaggeration and Hyperbole - Exaggerated claims to make information seem more significant or alarming.
    3. Bias and Subjectivity - Highly polarized language, presenting information in a biased manner.
    4. Repetition - Repeating keywords or phrases to reinforce the message.
    5. Specific Word Choices - Using complex jargon to lend false credibility.
    6. Appeals to Authority - Citing non-existent or unqualified authorities to support claims.
    7. Lack of Verifiable Sources - Lacking credible sources or referencing vague sources.
    8. Logical Fallacies - Using logical fallacies like straw man arguments or ad hominem attacks.
    9. Conspiracy Theories - Including elements of conspiracy theories.
    10. Inconsistencies - Containing contradictory statements within the same content.
    11. Disinformation - Deliberately misleading or biased information, manipulated narrative or facts intended to deceive.

    Please respond in the following format:
    - Emotional Appeal: [Present/Absent]
    - Exaggeration and Hyperbole: [Present/Absent]
    - Bias and Subjectivity: [Present/Absent]
    - Repetition: [Present/Absent]
    - Specific Word Choices: [Present/Absent]
    - Appeals to Authority: [Present/Absent]
    - Lack of Verifiable Sources: [Present/Absent]
    - Logical Fallacies: [Present/Absent]
    - Conspiracy Theories: [Present/Absent]
    - Inconsistencies: [Present/Absent]
    - Disinformation: [Likely/Unlikely]
    - Reasoning: [A brief reasoning if it is disinformation.]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are to analyze text for rhetorical techniques and disinformation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 
        )
        
        if response.choices:
            message_content = response.choices[0].message.content.strip()
            print(message_content)
            return message_content
        else:
            return None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def load_data():
    try:
        return pd.read_csv('gpt_analysis_test.csv')
    except FileNotFoundError:
        return pd.DataFrame()

def save_checkpoint(index):
    with open('checkpoint.txt', 'w') as file:
        file.write(str(index))

def load_checkpoint():
    try:
        with open('checkpoint.txt', 'r') as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def extract_reasoning(analysis_result):
    reasoning = ""
    if "-Reasoning:" in analysis_result:
        reasoning = analysis_result.split("-Reasoning:")[1].strip()
    return reasoning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text for disinformation and rhetorical techniques.")
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file containing text data.')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key for authentication.')
    parser.add_argument('--output_file', type=str, default='gpt_analysis_test.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()

    # Load the data from the CSV file
    data = pd.read_csv(args.input_file)
    data.columns = [col.strip('- ').replace(' ', '_') for col in data.columns]

    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_api_key)

    saved_data = load_data()
    start_index = load_checkpoint()
    results_list = []

    for index, row in data.iloc[start_index:].iterrows():
        text = row['first_paragraph']
        analysis_result = analyze_and_label_text(text, client)
        if analysis_result:
            reasoning = extract_reasoning(analysis_result)
            result = {
                **row.to_dict(),
                "analysis_result": analysis_result,
                "reasoning": reasoning
            }
            results_list.append(result)
            if len(results_list) >= 5:  # Save after processing 5 records
                batch_df = pd.DataFrame(results_list)
                saved_data = pd.concat([saved_data, batch_df], ignore_index=True)
                saved_data.to_csv(args.output_file, index=False)
                results_list = []  # Reset the results list after saving
        save_checkpoint(index + 1)

    # Save any remaining data after the loop completes
    if results_list:
        batch_df = pd.DataFrame(results_list)
        saved_data = pd.concat([saved_data, batch_df], ignore_index=True)
        saved_data.to_csv(args.output_file, index=False)
    print(f"Analysis complete. Results saved to {args.output_file}.")











