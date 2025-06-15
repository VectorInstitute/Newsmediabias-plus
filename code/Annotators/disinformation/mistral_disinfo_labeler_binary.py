from datetime import datetime
import pandas as pd
import os
import logging
import asyncio
import argparse
from openai import OpenAI
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)

async def analyze_and_label_text(text, client, model_id):
    """
    Analyze the text for disinformation using rhetorical techniques and return 'Likely' or 'Unlikely'.
    """
    prompt = f"""
        Assess the text below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.
        If you find any of the listed rhetorical techniques, then the article is likely disinformation; if not, it is likely not disinformation.
        Respond only with 'Likely' or 'Unlikely' without any further explanation.

        Text: {text}

        Rhetorical Techniques Checklist:
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

        Response format required: [Likely/Unlikely]
    """

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt} 
            ]
        )

        if response.choices:
            message_content = response.choices[0].message.content.strip()
            return message_content      
        else:
            return "No trained response."
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def convert_analysis_to_dataframe(row, disinformation_status):
    results = row.to_dict()
    results['Disinformation'] = disinformation_status
    return results

async def analyze_and_label_text_batch(batch_texts):
    tasks = [analyze_and_label_text(text) for text in batch_texts]
    return await asyncio.gather(*tasks)

async def main(args):
    # Specify model name and dataset split
    model_name = "Mistral-7B"
    data_split = args.data_split

    output_file = f"{model_name}_binarylabel_{data_split}.csv"
    checkpoint_file = f"processed_indices_{model_name}_{data_split}.txt"

    # OpenAI client setup
    client = OpenAI(base_url="http://gpu015:8080/v1", api_key="EMPTY")
    model_id = "/model-weights/Mistral-7B-Instruct-v0.3"


    data = pd.read_csv(f"annotations/{data_split}.csv")


    processed_indices = set()
    try:
        with open(checkpoint_file, 'r') as file:
            processed_indices = {int(line.strip()) for line in file}
        print("Loaded processed indices successfully.")
    except FileNotFoundError:
        print("No previous checkpoint file found, starting from scratch.")

    results_list = []
    batch_texts = []
    batch_indices = []
    batch_size = 10  

    total_rows = len(data)
    processed_rows = len(processed_indices)

    progress_bar = tqdm(total=total_rows, initial=processed_rows, desc="Processing batches")

    for index, row in data.iterrows():
        if index not in processed_indices:
            text = row.get('first_paragraph', '')
            if text:
                batch_texts.append(text)
                batch_indices.append(index)

                if len(batch_texts) >= batch_size:
                    start_time = time.perf_counter()

                    disinformation_results = await analyze_and_label_text_batch(batch_texts, client, model_id)

                    for i, disinformation_status in enumerate(disinformation_results):
                        if disinformation_status:
                            result = convert_analysis_to_dataframe(data.iloc[batch_indices[i]], disinformation_status)
                            results_list.append(result)

                    with open(checkpoint_file, 'a') as file:
                        for i in batch_indices:
                            file.write(f"{i}\n")
                    print(f"Saved checkpoint for batch up to index {index}.")

                    if results_list:
                        temp_df = pd.DataFrame(results_list)
                        temp_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                        results_list.clear()

                    progress_bar.update(len(batch_indices))

                    batch_time = time.perf_counter() - start_time
                    progress_bar.set_postfix(batch_time=f"{batch_time:.2f}s")

                    batch_texts.clear()
                    batch_indices.clear()

                    await asyncio.sleep(1)  

    if batch_texts:
        disinformation_results = await analyze_and_label_text_batch(batch_texts, client, model_id)

        for i, disinformation_status in enumerate(disinformation_results):
            if disinformation_status:
                result = convert_analysis_to_dataframe(data.iloc[batch_indices[i]], disinformation_status)
                results_list.append(result)

        with open(checkpoint_file, 'a') as file:
            for i in batch_indices:
                file.write(f"{i}\n")
        print("Saved remaining processed indices.")

        progress_bar.update(len(batch_indices))

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        print(f"Final results saved to '{output_file}'")
    else:
        print("No results were processed. Check your dataset and retry.")

    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and label text for disinformation.")
    parser.add_argument("--data_split", type=str, default="train", help="Dataset split to process (train/test).")
    args = parser.parse_args()
    asyncio.run(main(args))
