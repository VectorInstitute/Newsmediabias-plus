import pandas as pd
import time
import os
from openai import OpenAI
import asyncio
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "Phi-3-medium-128k-instruct"

# OpenAI client setup
client = OpenAI(base_url="http://gpu044:8080/v1", api_key="EMPTY")

def detect_bias_sync(article, model):
    """ Synchronously detect bias and extract explanation from the OpenAI API response. """
    print(f"Processing: {article[:30]}...")
    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""
                                You are a helpful news fact-checking bot trained to assess the accuracy of information. Your task is to analyze the given article and determine whether it is 'Factually Correct' or 'Factually Incorrect'. 

                                Fact-checking is the methodical process of verifying claims in public discourse or media reports. It is vital for countering misinformation and disinformation, thereby enhancing public knowledge and trust. Consider the following in your evaluation:

                                Misinformation: Incorrect or misleading information shared without intent to harm.
                                Disinformation: Information that is knowingly false, often prejudiced, and disseminated with the intent to mislead.

                                Your analysis should include:

                                Verification of key claims against multiple reliable sources.
                                Identification of logical fallacies or statements that may mislead readers.
                                Assessment of the context in which the information was presented, including the source’s history and potential motivations.
                                Evaluation for any presence of hate speech, linguistic harm, or intent to spread prejudice.
                                Provide your assessment in the following format:

                                Classification: [Factually Correct/Factually Incorrect]
                                Explanation: Provide a concise, evidence-based explanation for your classification. Reference specific examples from the article and contradicting evidence from trusted sources, if applicable.

                                Ensure to remain objective, basing your assessment strictly on facts and evidence rather than personal opinions or biases.

                                Article to analyze:
                                {article}
                                """
                }
            ]
        )
        response = completion.choices[0].message.content
        label = "Real" if "Factually Correct" in response else "Fake"

        # Extract and clean the explanation
        explanation = response.split('Explanation: ')[1] if 'Explanation: ' in response else "No explanation provided."
        explanation = explanation.replace('\n', ' ').replace('\t', ' ').strip()
        explanation = ' '.join(explanation.split())  # Removes redundant spaces and makes it a single line

        return label, explanation
    except Exception as e:
        logging.error("Failed to process article with error: %s", e)
        return "Error", "Failed to retrieve data"

async def detect_bias(article, model):
    """ Asynchronous wrapper to call the synchronous detect_bias_sync function """
    loop = asyncio.get_running_loop()  # Get the current running loop
    return await loop.run_in_executor(None, detect_bias_sync, article, model)

async def create_completion(row):
    """ Asynchronously process each row, extracting both label and explanation. """
    article = row['text_content_summary']
    label, explanation = await detect_bias(article, MODEL_NAME)
    
    result = [row['unique_id'],
            row['title'],
            row['text_content_summary'],             
        label, explanation
    ]
    print("results: ", result)
    return result

async def send_requests(data_batch):
    """ Send multiple asynchronous requests to process data batch. """
    tasks = [create_completion(row) for _, row in data_batch.iterrows()]
    results = await asyncio.gather(*tasks)
    return results

def save_results(results, output_file='phi3_label.csv'):
    """ Save CSV results, including explanations. """
    results_df = pd.DataFrame(results, columns=['unique_id', 'title','summary','label_phi','explanation_phi'
])
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file, mode='w', header=True, index=False)

if __name__ == "__main__":
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="Process news articles for bias detection using OpenAI's Phi-3 model.")
    parser.add_argument('--data_path', type=str, default='phi_500.csv', help='Path to the input CSV file containing news articles.')
    args = parser.parse_args()

    data_path = args.data_path
    data = pd.read_csv(data_path)  # Sample data for testing, adjust as necessary

    batch_size = 8
    num_batches = (len(data) + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(num_batches):
        prompts = data.iloc[i * batch_size:(i + 1) * batch_size]
        if not prompts.empty:
            results = asyncio.run(send_requests(prompts))
            save_results(results)
            print(f"Batch {i + 1}/{num_batches} processed and saved.")
            elapsed_time = time.time() - start_time
            print(f"Total time taken so far: {elapsed_time // 3600} hours, {(elapsed_time % 3600) // 60} minutes, {elapsed_time % 60} seconds")
            # time.sleep(4)

    print("All batches processed successfully.")
