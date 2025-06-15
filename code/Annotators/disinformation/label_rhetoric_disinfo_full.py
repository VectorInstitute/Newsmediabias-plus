import time
import os
import pandas as pd
import argparse
from openai import OpenAI


def analyze_and_label_text(text, client):
    """
    Analyze the text for disinformation using rhetorical techniques and return a structured response.
    """
    prompt = f"""
        Assess the text below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.
        If you find any of the listed rhetorical techniques, then the article is likely disinformation; if not, it is likely not disinformation.
s
        Text: {text}

    Stringent Rhetorical Techniques to Look for:
    - Emotional Appeal: Uses language that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.
    - Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.
    - Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.
    - Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.
    - Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.
    - Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.
    - Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.
    - Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.
    - Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.
    - Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.
    - Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.
    - Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.

        Format your response as:
        - Reasoning: [Provide a brief one line reasoning for your label.]
        - Disinformation: [Likely/Unlikely]

        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze text for disinformation based on the listed techniques."},
                {"role": "user", "content": prompt}
            ]
        )

        if response.choices:
            message_content = response.choices[0].message.content.strip()  # Corrected here
            print(message_content)
            return message_content      
        else:
             return "No testid response."
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def convert_analysis_to_dataframe(row, analysis_str):
    """
    Convert the analysis string into a dictionary format suitable for DataFrame.
    """
    results = row.to_dict()
    results['Disinformation'] = ''
    results['Reasoning'] = ''

    for line in analysis_str.split('\n'):
        if ': ' in line:
            category, detail = line.split(': ', 1)
            results[category.strip()] = detail.strip()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text for disinformation using rhetorical techniques.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing text data.')
    parser.add_argument('--output_file', type=str, default='gpt4o-test-binary.csv', help='Path to save the output CSV file.')
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key for authentication.')
    # Parse command line arguments
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.input_file)

    client = OpenAI(api_key=args.openai_api_key)

    # Checkpointing: Load processed indices if exists
    processed_indices = set()
    try:
        with open('processed_indices.txt', 'r') as file:
            processed_indices = {int(line.strip()) for line in file}
        print("Loaded processed indices successfully.")
    except FileNotFoundError:
        print("No previous checkpoint file found, starting from scratch.")

    
    results_list = []
    batch_indices = []



    for index, row in data.iterrows():
        if index not in processed_indices:
            text = row.get('first_paragraph', '')
            if text:
                analysis_result = analyze_and_label_text(text, client)
                if analysis_result:
                    result = convert_analysis_to_dataframe(row, analysis_result)
                    results_list.append(result)
                    batch_indices.append(index)
                    
                    if len(batch_indices) >= 10:
                        # Save processed indices
                        try:
                            with open('processed_indices.txt', 'a') as file:
                                for i in batch_indices:
                                    file.write(f"{i}\n")
                            print(f"Saved checkpoint for batch up to index {index}.")
                        except Exception as e:
                            print(f"Failed to write to file: {str(e)}")
                        batch_indices.clear()  # Clear the list after saving
                        
                        # Save results to CSV
                        try:
                            temp_df = pd.DataFrame(results_list)
                            temp_df.to_csv(args.output_file, mode='a', header=not os.path.exists(args.output_file), index=False)
                            results_list.clear()  # Clear the list after saving
                            print(f"Partial results saved to '{args.output_file}'")
                        except Exception as e:
                            print(f"Failed to save partial results to CSV: {str(e)}")

                    time.sleep(1)  # Respect API rate limits

    

    # Save any remaining processed indices
    if batch_indices:
        with open('processed_indices.txt', 'a') as file:
            for i in batch_indices:
                file.write(f"{i}\n")
        print("Saved remaining processed indices.")


    # Save any remaining results to CSV
    if results_list:
        try:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(args.output_file, mode='a', header=not os.path.exists(args.output_file), index=False)
            print(f"Final results saved to '{args.output_file}'")
        except Exception as e:
            print(f"Failed to save final results to CSV: {str(e)}")
    else:
        print("No results were processed. Check your dataset and retry.")