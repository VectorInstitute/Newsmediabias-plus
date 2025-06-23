import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import csv
import uuid
import pandas as pd
import os

# This script uses the LLama3 model to classify bias in text articles.

models = {}

def load_model(model_name):
    """
    Load the specified model and tokenizer if not already loaded.
    """
    if model_name not in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype=torch.bfloat16, 
                                                     device_map="auto")
        models[model_name] = pipeline("text-generation", 
                                      model=model, 
                                      tokenizer=tokenizer)
    return models[model_name]

def truncate_to_5000_words(text):
    """
    Truncate the text to the first 5000 words.
    """
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    return ' '.join(words[:5000]) if len(words) > 5000 else cleaned_text

def detect_bias(model_name, prompt, article):
    """
    Detect bias in the given article using the specified model and prompt.
    Args:
        model_name (str): Name of the model to use for classification.
        prompt (str): Prompt to guide the model in classification.
        article (str): The article text to classify.
    Returns:
        tuple: A tuple containing the label and a list of biased words/phrases.
    """
    pipe = load_model(model_name)
    
    full_prompt = f"{prompt}\n\nArticle: {article}\n\nClassification:"
    response = pipe(full_prompt, 
                    max_new_tokens=200, 
                    do_sample=True, 
                    temperature=0.7)[0]['generated_text']
    
    label = "Unbiased"
    biased_words = []
    if "<|LABEL_SEP|>" in response:
        parts = response.split("<|LABEL_SEP|>")
        label_section = parts[0]
        biased_words_section = parts[1] if len(parts) > 1 else ""
        if "heavily biased" in label_section.lower():
            label = "Heavily Biased"
        elif "moderately biased" in label_section.lower():
            label = "Moderately Biased"
        elif "slightly biased" in label_section.lower():
            label = "Slightly Biased"
        if "Biased words/phrases:" in biased_words_section:
            biased_words_section = biased_words_section.split("Biased words/phrases:")[1]
            biased_words = biased_words_section.strip().split("<|SEPARATOR|>")
    
    return label, biased_words

def process_and_save(model_name, prompt, text_input, file_input, output_file):
    """
    Process the input text or file, classify bias, and save results to a CSV file.
    Args:
        model_name (str): Name of the model to use for classification.
        prompt (str): Prompt to guide the model in classification.
        text_input (str): Manual text input for classification.
        file_input (file): Optional CSV file input with articles.
        output_file (str): Name of the output CSV file to save results.
    Returns:
        str: Confirmation message indicating the number of processed entries and output file location.
    """
    results = []
    
    if file_input:
        df = pd.read_csv(file_input.name, encoding='utf-8')
        for _, row in df.iterrows():
            article = truncate_to_5000_words(row['text_content'])
            label, biased_words = detect_bias(model_name, 
                                              prompt, 
                                              article)
            unique_id = str(uuid.uuid4())
            results.append([unique_id, 
                            article, 
                            label, 
                            ', '.join(biased_words)])
    else:
        article = truncate_to_5000_words(text_input)
        label, biased_words = detect_bias(model_name, 
                                          prompt, 
                                          article)
        unique_id = str(uuid.uuid4())
        results.append([unique_id, 
                        article, 
                        label, 
                        ', '.join(biased_words)])
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.path.getsize(output_file) == 0:
            writer.writerow(['Unique ID', 
                             'Article', 
                             'Label', 
                             'Biased Words/Phrases'])
        writer.writerows(results)
    
    return f"Processed {len(results)} entries. Results saved to {output_file}"

with gr.Blocks() as demo:
    gr.Markdown("# Text Bias Classification Pipeline")
    
    with gr.Row():
        model_name = gr.Dropdown(
            label="Model Name",
            choices=["meta-llama/Meta-Llama-3-8B-Instruct", 
                     "meta-llama/Meta-Llama-3.1-8B-Instruct"],
            value="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            value="You are a news article classifier bot. The news article can be classified into the following categories: 'Unbiased', 'Slightly Biased', 'Moderately Biased', 'Heavily Biased'. Your task is to read the following articles and classify them using the above categories. Additionally, please list biased words if any found in the article, separated by '<|SEPARATOR|>'."
        )
    
    with gr.Row():
        text_input = gr.Textbox(label="Text to Classify (Manual Input)", 
                                lines=5)
        file_input = gr.File(label="CSV File Input (Optional)")
    
    output_file = gr.Textbox(label="Output File Name", 
                             value="bias_classifications.csv")
    
    classify_button = gr.Button("Classify and Save")
    
    output = gr.Textbox(label="Result", 
                        lines=4)
    
    classify_button.click(
        process_and_save,
        inputs=[model_name, 
                prompt, 
                text_input, 
                file_input, 
                output_file],
        outputs=output
    )

demo.launch(share=True)
