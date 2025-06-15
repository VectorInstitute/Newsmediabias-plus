import pandas as pd
import time
import os
import logging
import asyncio
import argparse
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)


def analyze_and_label_text(text, client, model_id="/model-weights/Mistral-Large-Instruct-2407"):
    """
    Analyze the text for rhetorical techniques and disinformation using the OpenAI API.
    Args:
        text (str): The text to analyze.
        client (OpenAI): The OpenAI client instance.
        model_id (str): The model ID to use for analysis.
    Returns:
        str: The analysis result indicating the presence of rhetorical techniques and disinformation likelihood.
    """
    prompt = f"""
    You are to analyze the given text for the presence of specific rhetorical techniques and the likelihood of disinformation. Indicate whether each rhetorical technique is present or absent, and state if disinformation is likely or unlikely.

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

    Please respond in the following format without giving details:
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
    """

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
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

def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return pd.DataFrame()

def save_checkpoint(index, output_dir):
    with open(f'{output_dir}/checkpoint.txt', 'w') as file:
        file.write(str(index))

def load_checkpoint(output_dir):
    try:
        with open(f'{output_dir}/checkpoint.txt', 'r') as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def extract_reasoning(analysis_result):
    reasoning = ""
    if "- Reasoning:" in analysis_result:
        reasoning = analysis_result.split("- Reasoning:")[1].strip()
    return reasoning

async def process_texts(data, saved_data, start_index, input_file, output_dir, client, model_id="/model-weights/Mistral-Large-Instruct-2407"):
    """
    Process texts in batches, analyze them, and save results to a CSV file.
    Args:
        data (pd.DataFrame): DataFrame containing the texts to analyze.
        saved_data (pd.DataFrame): DataFrame to store results.
        start_index (int): Index to start processing from.
    """
    results_list = []
    tasks = []
    batch_size = 8  
    num_batches = (len(data) - start_index) // batch_size + 1
    start_time = time.time()

    for index, row in data.iloc[start_index:].iterrows():
        text = row['first_paragraph']
        task = asyncio.ensure_future(asyncio.get_event_loop().run_in_executor(None, analyze_and_label_text, text, client, model_id))
        tasks.append((task, index, row))
        
        # Process in batches
        if len(tasks) >= batch_size:
            responses = await asyncio.gather(*[t[0] for t in tasks])
            for response, (task, index, row) in zip(responses, tasks):
                if response:
                    reasoning = extract_reasoning(response)
                    result = {
                        **row.to_dict(),
                        "analysis_result": response,
                        "reasoning": reasoning
                    }
                    results_list.append(result)
                    save_checkpoint(index + 1, output_dir)

            # Save batch
            if results_list:
                batch_df = pd.DataFrame(results_list)
                saved_data = pd.concat([saved_data, batch_df], ignore_index=True)
                saved_data.to_csv(f'{output_dir}/{input_file}', index=False)
                results_list = []  # Reset the results list after saving
            tasks = []  # Reset tasks after processing a batch

            # Calculate batch number
            current_batch = (index - start_index) // batch_size + 1
            print(f"Batch {current_batch}/{num_batches} processed and saved.")
            elapsed_time = time.time() - start_time
            print(f"Total time taken so far: {elapsed_time // 3600} hours, {(elapsed_time % 3600) // 60} minutes, {elapsed_time % 60} seconds")

    # Process any remaining tasks after the loop
    if tasks:
        responses = await asyncio.gather(*[t[0] for t in tasks])
        for response, (task, index, row) in zip(responses, tasks):
            if response:
                reasoning = extract_reasoning(response)
                result = {
                    **row.to_dict(),
                    "analysis_result": response,
                    "reasoning": reasoning
                }
                results_list.append(result)
                save_checkpoint(index + 1, output_dir)

    # Save any remaining data after all processing
    if results_list:
        batch_df = pd.DataFrame(results_list)
        saved_data = pd.concat([saved_data, batch_df], ignore_index=True)
        saved_data.to_csv(f'{output_dir}/{input_file}', index=False)

async def main(args):
    start_time = time.time()  # Start timing here 
    
    # Load the data
    data = pd.read_csv(args.data_file)
    data.columns = [col.strip('- ').replace(' ', '_') for col in data.columns]
    
    saved_data = load_data(args.input_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    start_index = load_checkpoint(args.output_dir)


    # OpenAI client setup
    # client = OpenAI(base_url="http://gpu042:8080/v1", api_key="EMPTY")
    # model_id = "/model-weights/gemma-2-9b-it"

    client = OpenAI(base_url="http://172.17.8.10:8081/v1", api_key="EMPTY")
    model_id = "/model-weights/Mistral-Large-Instruct-2407"

    await process_texts(data, saved_data, start_index, args.input_file, args.output_dir, client, model_id)
    
    # Calculate total time after all processing is complete
    total_time = time.time() - start_time
    logging.info(f'Total processing time: {total_time:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text data for disinformation analysis.')
    parser.add_argument('--data_file', type=str, default='train.csv', help='Path to the input CSV file')
    parser.add_argument('--input_file', type=str, default='mistrallarge_analysis.csv', help='Input CSV file with text data')
    parser.add_argument('--output_dir', type=str, default='re-label-results', help='Directory to save results')
    args = parser.parse_args()
    asyncio.run(main(args))
