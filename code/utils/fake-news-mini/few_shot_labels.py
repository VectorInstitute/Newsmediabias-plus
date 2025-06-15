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

MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"

# OpenAI client setup
client = OpenAI(base_url="http://gpu013:8080/v1", api_key="EMPTY")

# Define 5-shot examples
examples = [
    {
        "article": "Former congressional aide Celeste Maloy will win the Republican special primary election to succeed her onetime boss, GOP Rep. Chris Stewart. Stewart is expected to vacate the seat on September 15 over his wife's health concerns. Maloy is also poised to be the first woman in Utah’s congressional delegation since Republican Mia Love left office in 2019. She will next face Democratic state Sen. Kathleen Riebe, who will be a decided underdog in the November 21 general election for a district that covers the Salt Lake City area to St. George. Stewart won a sixth term last fall by 26 points, while former President Donald Trump would have carried it under its current lines by 17 points in 2020, according to a CNN/ORC poll.",
        "label": "Factually Correct",
        "explanation": "The article presents several claims that can be verified against multiple reliable sources: 1. Celeste Maloy's expected win in the Republican special primary election: Although the article does not provide a specific source, the claim can be verified by checking the official results of the election or reputable news sources that covered the event. 2. Rep. Chris Stewart's expected resignation due to his wife's health concerns: This claim is also unverified, but it is consistent with Stewart's subsequent resignation on September 15, which was reported by multiple news sources. 3. Celeste Maloy's potential to be the first woman in Utah's congressional delegation since Mia Love: This claim can be verified by checking the official congressional directories or reputable sources that track the composition of Congress. 4. The expected outcome of the general election between Celeste Maloy and Democratic state Sen. Kathleen Riebe: The article's characterization of Riebe as a `decided underdog' is consistent with the district's Republican lean, as evidenced by Stewart's 26-point victory in the previous election and the district's 17-point margin for President Trump in 2020, according to the cited CNN/ORC poll. However, the article does not provide any sources to support these claims, and the information could be considered speculative or unverified. Nevertheless, based on the information available, the claims appear to be factually correct."
    },
    {
        "article": "The U.S. is likely to be in a pickle no matter who wins in Taiwan's presidential election. The key issue boils down to Taiwan’s tumultuous relationship with China. Current Taiwan Vice President Lai Ching-te of the DPP is narrowly ahead in the polls. His party supports Taiwanese sovereignty and identity. The other looming election to watch is that of the U.K. presidential election on November 8, 2015, which is expected to be a close race. The winner of that election will be the next president of the United States, which will be inaugurated on January 20, 2016, if the current president is not re-elected in 2016. The current president, Barack Obama, is seeking a second term.",
        "label": "Factually Incorrect",
        "explanation": "The article contains several inaccuracies and misleading statements. Here's a breakdown of the issues: 1. **Taiwan's presidential election:** The article does not specify which election it is referring to, but Taiwan's president is elected every four years. The most recent election was in 2020, and the next one is scheduled for 2024. The article's mention of a presidential election without specifying the year or date is vague and potentially misleading. 2. **U.K. presidential election:** The article claims that a U.K. presidential election is taking place on November 8, 2015. However, there is no such election scheduled for that date in the United Kingdom. The UK has a parliamentary system, and the head of government is the Prime Minister, not a president. The UK does not have a direct presidential election. 3. **U.S. presidential election:** The article implies that the winner of the U.K. non-existent presidential election will become the next president of the United States. This is not accurate. The 2016 U.S. presidential election was held on November 8, 2016, and Donald Trump won the election, not the `winner of the U.K. presidential election.' 4. **Barack Obama's term:** The article claims that Barack Obama is seeking a second term in 2016. While Obama was re-elected in 2012, he was not seeking a second term in 2016, as his second term had already begun in 2013. Additional issues: * The article uses the term `presidential election' for both Taiwan and the UK, which can be confusing, especially for readers unfamiliar with the different electoral systems. * The article's tone is sensationalized, using phrases like `pickle' and `looming election,' which may be intended to attract attention rather than provide accurate information. In conclusion, the article contains multiple factual errors and misleading statements, rendering it Factually Incorrect."
    },
    {
        "article": "Darling served as chancellor under Gordon Brown from 2007 to 2010. He was an Edinburgh MP from 1987 until he stepped down from parliament in 2015. Brown said he was ‘deeply saddened’ by news of Darling’s death. Labour leader, Keir Starmer, said: ‘Alistair lived a life devoted to public service. He will be remembered as the chancellor.’ He died after a short spell in Western General hospital under the wonderful care of the cancer team, a spokesperson for the family said. He is survived by his wife Margaret and their two children, Calum and Anna. He leaves behind a wife, Margaret, and two daughters, Anna and Calum, as well as a son, Andrew.",
        "label": "Factually Correct",
        "explanation": "The article appears to be a factual obituary, and after verifying the key claims against multiple reliable sources, I found that the information is accurate. * Alistair Darling was indeed the Chancellor of the Exchequer under Gordon Brown from 2007 to 2010, as confirmed by the UK Parliament's website and the BBC. * Darling was an MP for Edinburgh South from 1987 to 2015, as per the UK Parliament's website and various news outlets. * Gordon Brown and Keir Starmer's statements about Darling's passing and legacy are likely quotes from genuine sources, but I couldn't verify their exact wording or context. However, their sentiments are consistent with previous tributes to Darling. * The article's description of Darling's passing and his family members is consistent with multiple reports from reputable news sources, including the BBC and The Guardian. However, I did notice a minor inconsistency in the article's description of Darling's family members. According to the article, Darling leaves behind a wife, Margaret, and two daughters, Anna and Calum, as well as a son, Andrew. However, the spokesperson for the family is quoted as saying that Darling is survived by his wife Margaret and their two children, Calum and Anna. This discrepancy is minor and likely a result of the family spokesperson's mistake or a misquote. Nevertheless, it is worth noting that the article's overall factual accuracy is high. I couldn't find any evidence of misinformation, disinformation, hate speech, linguistic harm, or intent to spread prejudice in the article. The language used is respectful and mournful, reflecting the passing of a notable public figure."
    },
    {
        "article": "RALEIGH, NC — At a rally following a disastrous debate performance, a guy who has spent the last few years imprisoning his political opponents warned the crowd that if Donald Trump is elected, he will imprison his political opponents.  The frightening accusation that Trump will use the power of the federal government to persecute his enemies seemed to have a significant impact coming from the man who has gained extensive experience in using the power of the federal government to persecute his most prominent enemies.  `This is serious business, folks,' Biden said to his assembled supporters. `The idea… that someone would use the power of… the power… their office… to seek revenge on their political opponents goes against everything… goes against all of the… goes… goes against… is an attempt to destroy democracy. That is why I have spent the last three years deploying the Justice Department against Donald Trump to put him in prison. Larbinffermonder.",
        "label": "Factually Inorrect",
        "explanation": "The article contains several inaccuracies and misleading statements. Firstly, the article incorrectly quotes Joe Biden as saying `Larbinffermonder' which does not appear to be a real word and seems to be a made-up phrase used to mock Biden's speaking style. Secondly, the article claims that Biden has spent the last three years imprisoning his political opponents, which is a false statement. While Biden has been the President since 2021, he has not imprisoned any political opponents. Thirdly, the article implies that Biden is hypocritical for accusing Trump of using the power of the federal government to persecute his enemies, while Biden himself has allegedly used the Justice Department to imprison his opponents. This is a misleading statement, as there is no evidence to suggest that Biden has imprisoned any political opponents. Furthermore, the article's tone and language are sensationalized and biased, which may be an attempt to mislead readers. It is essential to note that the article's accuracy is further compromised by the lack of specific examples or evidence to support the claims made about Biden's actions. In contrast, reputable news sources, such as CNN, The New York Times, and The Washington Post, have reported on Biden's presidency and policies without making such unfounded claims. In conclusion, the article contains several factual inaccuracies and misleading statements, which classify it as Factually Incorrect."
    },
    {
        "article": "New Delhi CNN —  Indians around the world have united with pride and excitement after the South Asian nation on Wednesday became the first country to soft land a spacecraft on the moon’s unexplored South Pole, and only the fourth country to reach the lunar surface.  A lander named Vikram and a rover called Pragyan, robots from India’s Chandrayaan-3 mission, touched down on the moon just after 6 p.m. local time – a moment watched by 70 million people on the Indian Space Research Organization (ISRO)’s YouTube page.",
        "label": "Factually Correct",
        "explanation": "The article reports that India's Chandrayaan-3 mission successfully landed a lander named Vikram and a rover called Pragyan on the moon's South Pole. This information is verified against multiple reliable sources, including: * The Indian Space Research Organization (ISRO) confirmed the successful landing on their website and social media channels. * NASA's website, which has a strong track record of verifying spacecraft missions, also reported the achievement. * Multiple reputable news outlets, such as BBC News, The New York Times, and The Guardian, published articles confirming the successful landing. The article's claim that India is the `first country to soft land a spacecraft on the moon's unexplored South Pole' is also accurate, according to ISRO and other reliable sources. However, it's worth noting that the article may be slightly misleading by stating that India is `only the fourth country to reach the lunar surface.' While this is true, it's not entirely clear what the article means by `lunar surface.` If it is referring to the South Pole specifically, then India is not the fourth country to reach the lunar surface, as the article claims. But if it's referring to the lunar surface in general, then India is indeed one of the few countries to have achieved this feat. Contextually, the article appears to be a factual report from a reputable news source, and there is no evidence of hate speech, linguistic harm, or intent to spread prejudice. The source, CNN, is a well-established and respected news organization. Overall, the article appears to be an accurate report of a significant scientific achievement, and its classification as Factually Correct is supported by the evidence."
    }
]

def create_5_shot_prompt(article):
    """Generate a prompt incorporating 5-shot examples and the article to analyze."""
    example_prompt = ""
    for ex in examples:
        example_prompt += f"""
                            Article: {ex['article']}
                            Classification: {ex['label']}
                            Explanation: {ex['explanation']}\n
                            """

    return f"""
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
            Here are some examples:

            {example_prompt}

            Now, analyze the following article:

            Article: {article}

            """

def detect_bias_sync(article, model):
    """ Synchronously detect bias and extract explanation from the OpenAI API response. """
    print(f"Processing: {article[:30]}...")
    try:
        prompt = create_5_shot_prompt(article)
        completion = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
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
    
    result = [
        row['unique_id'], row['title'], article,
        label, explanation
    ]
    print("results: ", result)
    return result

async def send_requests(data_batch):
    """ Send multiple asynchronous requests to process data batch. """
    tasks = [create_completion(row) for _, row in data_batch.iterrows()]
    results = await asyncio.gather(*tasks)
    return results


def save_results(results, filename):
    """ Save CSV results, including explanations. """
    results_df = pd.DataFrame(results, columns=['unique_id', 'title', 'text_content_summary', 'label_llama3.1_summary', 'explanation_llama3.1_summary'])
    if os.path.exists(filename):
        results_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        results_df.to_csv(filename, mode='w', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process news articles for fact-checking.")
    parser.add_argument('--data_path', type=str, default='train_llama3.csv', help='Path to the input CSV file containing news articles.')
    parser.add_argument('--output_path', type=str, default='Llama3.1_5shot.csv', help='Path to save the output CSV file with results.')
    args = parser.parse_args()

    data_path = args.data_path
    data = pd.read_csv(data_path) 

    batch_size = 8
    num_batches = (len(data) + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(num_batches):
        prompts = data.iloc[i * batch_size:(i + 1) * batch_size]
        if not prompts.empty:
            results = asyncio.run(send_requests(prompts))
            save_results(results, filename=args.output_path)
            print(f"Batch {i + 1}/{num_batches} processed and saved.")
            elapsed_time = time.time() - start_time
            print(f"Total time taken so far: {elapsed_time // 3600} hours, {(elapsed_time % 3600) // 60} minutes, {elapsed_time % 60} seconds")
            # time.sleep(4)

    print("All batches processed successfully.")
