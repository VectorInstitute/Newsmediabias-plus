import os
import base64
import csv
import logging
import argparse
from openai import OpenAI
from pydantic import BaseModel
import json
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

image_extensions = [".jpg", ".jpeg", ".png"]

class ImageAnnotation(BaseModel):
    description: str
    bias_analysis: str
    label: str

PROMPT_TEMPLATE = """
<|user|>\n<|image_1|>
Given the image and any provided context from a news article, perform the following tasks and respond in a JSON format:

Context: {context}

Tasks:

Image Description: Provide a brief, objective sentence describing the main content of the image, including any relevant objects, people, actions, or events.

Key: description
Bias Analysis: Examine the image and any given context for potential negative aspects or biases by considering the following:

The tone or mood conveyed (positive, negative, or neutral)
Emotions the image might evoke and why
The balance of the depiction or any emphasis on negative aspects
Composition, framing, or other elements in the image that could influence perception Summarize your analysis in one sentence, focusing on the most significant aspect related to potential bias or negative portrayal.
Key: bias_analysis
Image Classification: Classify the image based on your analysis:

"Biased" if it appears to present a skewed or negative perspective
"Unbiased" if it seems to present a balanced or neutral view
Key: label
Response Format: Ensure your response is a concise JSON object containing the specified keys and their corresponding values.
<|end|>\n<|assistant|>\n
"""


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None
    
def read_csv_data(csv_path):
    """
    Read the CSV file and return a dictionary mapping unique IDs to first paragraphs.
    """
    data = {}
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return None

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            required_columns = {'unique_id', 'first_paragraph'}
            available_columns = set(reader.fieldnames)
            missing_columns = required_columns - available_columns

            if missing_columns:
                logging.error(f"CSV file {csv_path} is missing required columns: {', '.join(missing_columns)}")
                return None

            for row in reader:
                unique_id = row['unique_id']
                first_paragraph = row.get('first_paragraph', '')
                data[unique_id] = first_paragraph

        if not data:
            logging.warning(f"No valid data found in CSV file {csv_path}")
        return data

    except IOError as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return None
    except csv.Error as e:
        logging.error(f"CSV parsing error in file {csv_path}: {e}")
        return None
    
def annotate_image(image_path, context):
    """
    Annotate the image using OpenAI's GPT model with the provided context.
    Args:
        image_path (str): Path to the image file.
        context (str): Context or description related to the image.
    Returns:
        str: The annotation result in JSON format.
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    prompt = PROMPT_TEMPLATE.format(context=context)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        annotation = response.choices[0].message.content
        print(f"Annotation for {image_path}: {annotation}")  # Print annotation to console
        return annotation
    except Exception as e:
        logging.error(f"Error annotating image {image_path}: {e}")
        return None

def append_results(results, output_path):
    try:
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'unique_id', 'context', 'description', 'bias_analysis', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in results:
                try:
                    annotation = json.loads(result['annotation'])
                    writer.writerow({
                        'filename': result['filename'],
                        'unique_id': result['unique_id'],
                        'context': result['context'],
                        'description': annotation['description'],
                        'bias_analysis': annotation['bias_analysis'],
                        'label': annotation['label']
                    })
                except json.JSONDecodeError as je:
                    logging.error(f"JSON parsing error for {result['filename']}: {je}")
                except KeyError as ke:
                    logging.error(f"Missing key in annotation for {result['filename']}: {ke}")
        logging.info(f"Results appended to {output_path}")
    except IOError as e:
        logging.error(f"Error appending results to {output_path}: {e}")

def save_checkpoint(last_processed_file):
    with open('checkpoint.json', 'w') as f:
        json.dump({'last_processed_file': last_processed_file}, f)
               
def load_checkpoint():
    try:
        with open('checkpoint.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def process_images_in_folder(folder_path, csv_data, output_path):
    """
    Process all images in the specified folder, annotating them using the CSV data.
    Args:
        folder_path (str): Path to the folder containing images.
        csv_data (dict): Dictionary mapping unique IDs to context from the CSV file.
        output_path (str): Path to save the output CSV file with annotations.
    """
    results = []
    checkpoint = load_checkpoint()
    start_from = checkpoint['last_processed_file'] if checkpoint else None
    started = start_from is None

    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if not started:
            if filename == start_from:
                started = True
            else:
                continue

        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(folder_path, filename)
            unique_id = os.path.splitext(filename)[0]  # Remove file extension
            context = csv_data.get(unique_id, "No context available")
            logging.info(f"Processing {filename}...")
            annotation = annotate_image(image_path, context)
            if annotation:
                results.append({
                    'filename': filename,
                    'unique_id': unique_id,
                    'context': context,
                    'annotation': annotation
                })
            else:
                logging.warning(f"Failed to annotate {filename}")

            # Save every 5 records and update checkpoint
            if (i + 1) % 5 == 0:
                append_results(results, output_path)
                save_checkpoint(filename)
                results = []  # Clear results after saving

    # Save any remaining results
    if results:
        append_results(results, output_path)
        save_checkpoint(filename)

def signal_handler(signum, frame):
    logging.info("Interrupt received, stopping gracefully...")
    sys.exit(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Image Annotator")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--data_dir", type=str, default="consolidated_data",
                        help="Path to the directory containing images and CSV file.")
    parser.add_argument("--output_annotation_file", type=str, default="image_annotations.csv",)
    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", args.openai_key))

    global DATA_DIR
    DATA_DIR = args.data_dir


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    folder_path = os.path.join(DATA_DIR, "images")
    csv_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    output_path = args.output_annotation_file

    if not os.path.exists(folder_path):
        logging.error(f"Image folder not found: {folder_path}")
        exit(1)

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        exit(1)

    csv_data = read_csv_data(csv_path)
    if not csv_data:
        logging.error("No valid data found in CSV file.")
        exit(1)

    # Initialize output CSV with headers if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'unique_id', 'context', 'description', 'bias_analysis', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    process_images_in_folder(folder_path, csv_data, output_path)