import pandas as pd
import openai
from tqdm import tqdm
import argparse
import time
from openai import OpenAI
import os
from bias_entity import BiasEntityExtraction

def configure_openai_api():
    """Set up OpenAI API key and environment variable."""
    api_key = 'sk-proj-'  # ← Replace with your actual OpenAI API key
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)

def create_evaluation_prompt(text):
    """
    Create a prompt for evaluating bias in a given text using NER labeling.
    Args:
        text (str): The text to be evaluated for bias.
    Returns:
        str: The formatted prompt for the OpenAI API.
    """
    # disinformation_definition = (
    #     "You can identify any explicit or implicit bias in the given text by examining the following elements:\n\n"
    #     "- Loaded Language: Pay attention to words with strong emotional connotations, such as \"terrorist,\" \"illegal alien,\" or \"welfare queen.\".\n"
    #     "- Exaggeration and Hyperbole: Makes unsupported claims or presents ordinary situations as extraordinary to manipulate perceptions.\n"
    #     "- Stereotyping:Look for generalizations about entire groups of people, often based on race, ethnicity, gender, or religion..\n"
    #     "- Slanting: This involves presenting information in a way that favors one side of an issue.\n"
    #     "- Manipulative Word Choices: Uses emotionally charged or misleading terms to subtly influence opinions.\n"
    #     "- Target Audience: Understand who the intended audience is and how that might influence the content.\n"
    #     "- Lack of Verifiable Sources: Relies on unverifiable or fabricated sources to support claims.\n"
    #     "- Logical Fallacies: Engages in flawed reasoning (e.g., circular logic, personal attacks) that undermines logical debate.\n"
    #     "- Selective Reporting: Notice if the article focuses on certain aspects of a story while ignoring others.\n"
    #     "- Inconsistencies and Factual Errors: Contains contradictions or easily disprovable inaccuracies, showing disregard for truth.\n"
    #     "- Selective Omission: Deliberately leaves out crucial information that would provide a more balanced understanding.\n\n"
    # )
    user_content = """Analyze each 1-gram, 2-gram and 3-grams of an entire article for NER labelling of your given entity. 
        If any of the words in th n-grams in the article contain this given part-of-speech, it should be labeled with one of: B-BIAS, I-BIAS, O.
        Identify where, if at all, the entity should be labeled on the sentence, using BIO format:
            B- prefix indicates the beginning of an entity.
            I- prefix indicates the inside of an entity.
            O indicates a token outside any entity.
            Every word should have exactly one entity tag.
        If the given entity is not relevant to the n-gram or sentence in question, it should be labeled with "O".

        Also check that all entities are continuous (an O tag cannot be followed by an I tag). Do not generate anything other than the list."""
    prompt = (
        f"{user_content}.\n\n"
        # f"Definition of Bias:\n{disinformation_definition}\n\n"
        f"Input Text: {text}\n\n"
    )
    return prompt

def get_api_response(prompt):
    """
    Get the response from the OpenAI API for the given prompt.
    Args:
        prompt (str): The prompt to send to the OpenAI API.
    Returns:
        dict: The response from the OpenAI API.
    """
    system_content = """You are a helpful NER assistant, and your job is to take a news article, and return a list of bias labels ["B-BIAS", "I-BIAS", "O"] 
        (one for each word in the article)."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        n=1,
        stop=["\n\n"],
        max_tokens=2048,
        temperature=0.4,
        response_format=BiasEntityExtraction,
    )
    print("Prompt Tokens:", response['usage']['prompt_tokens'])
    print("Completion Tokens:", response['usage']['completion_tokens'])
    print("Total Tokens:", response['usage']['total_tokens'])
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER bias evaluation with OpenAI GPT API.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, default="NER-gpt-results.csv", help="Path to save output CSV.")
    parser.add_argument("--nrows", type=int, default=20, help="Number of rows to process (default: 20).")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv, nrows=args.nrows, usecols=["unique_id", "article_text", "text_label"])
    # df_biased = df[df['text_label'] == 'Biased']

    client = configure_openai_api()

    ner_results = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Rows"):
        # Swap this line if you want to use actual article text
        # text = row['article_text']
        text = "The survey is weighted to be representative of the U.S. adult population by gender, race, ethnicity, partisan affiliation, education and other categories."

        prompt = create_evaluation_prompt(text)

        try:
            response = get_api_response(client, prompt)
            evaluation = response.choices[0].message.content
            print(f"GPT Response for row {index}: {evaluation} - response length = {len(evaluation)}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            evaluation = "No explanation provided"

        ner_results.append(evaluation)
        time.sleep(1)  # Respect API rate limits

    # Save results
    df['evaluation'] = ner_results
    df.to_csv(args.output_csv, index=False)
    print(f"Evaluation complete. Results saved to '{args.output_csv}'.")