import os
import base64
import csv
import logging
from openai import OpenAI
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "Phi-3-vision-128k-instruct"

client = OpenAI(base_url="http://gpu049:8080/v1", api_key="EMPTY")
image_extensions = [".jpg", ".jpeg", ".png"]

def annotate_image(image_path, context):
    """
    Annotate an image using the Phi-3-vision-128k-instruct model.
    Args:
        image_path (str): Path to the image file.
        context (str): Context from the CSV file related to the image.
    Returns:
        str: Annotation result containing description, bias analysis, and bias label.
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    prompt = f"""
    Analyze the given image from a news article. Your response must follow this format:

    description:<description of the image> (Describe the image objectively in one sentence, including its content, relevant objects, people, actions, or events depicted.)
    bias_analysis:<analysis of the potential bias> (Analyze the image and provided context, focusing on potential negative aspects or biases. Consider, the overall tone or mood conveyed (positive, negative, or neutral), Emotions the image might evoke and why, Whether the image presents a balanced view or emphasizes negative aspects, Any notable elements in the image's composition, framing, or content that could influence perception, Summarize your analysis in one sentence, highlighting the most significant aspect related to potential bias or negative portrayal.)
    bias_label:<either "Biased" or "Unbiased"> (Based on your analysis, classify the image as either "Biased" if it appears to present a skewed or negative perspective, or "Unbiased" if it seems to present a balanced or neutral view.)

    Ensure that your response strictly adheres to this format.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error annotating image {image_path}: {str(e)}", exc_info=True)
        return None

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
            missing_columns = required_columns - set(reader.fieldnames)

            if missing_columns:
                logging.error(f"CSV file {csv_path} is missing required columns: {', '.join(missing_columns)}")
                return None

            for row in reader:
                unique_id = row['unique_id']
                first_paragraph = row.get('first_paragraph', '')
                data[unique_id] = first_paragraph

        return data
    except IOError as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return None

def write_results(results, output_path):
    try:
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 
                          'unique_id', 
                          'context', 
                          'description', 
                          'bias_analysis', 
                          'bias_label']
            writer = csv.DictWriter(csvfile, 
                                    fieldnames=fieldnames)
            for result in results:
                annotation = result['annotation']
                
                # Ensure annotation contains the expected keys
                if "description:" in annotation and "bias_analysis:" in annotation and "bias_label:" in annotation:
                    try:
                        # Extract content between the keys using strict boundaries
                        description_part = annotation.split('description:')[1].split('bias_analysis:')[0].strip().split('\n')[0]
                        bias_analysis_part = annotation.split('bias_analysis:')[1].split('bias_label:')[0].strip().split('\n')[0]
                        bias_label_part = annotation.split('bias_label:')[1].strip().split('\n')[0]

                        writer.writerow({
                            'filename': result['filename'],
                            'unique_id': result['unique_id'],
                            'context': result['context'],
                            'description': description_part,
                            'bias_analysis': bias_analysis_part,
                            'bias_label': bias_label_part
                        })
                    except Exception as ie:
                        logging.error(f"Error parsing response for {result['filename']}: {ie}")
                else:
                    logging.warning(f"Annotation for {result['filename']} did not contain the expected format: {annotation}")
        logging.info(f"Results written to {output_path}")
    except IOError as e:
        logging.error(f"Error writing results to {output_path}: {e}")

def process_image(image_info):
    """
    Process a single image and its associated context from the CSV file.
    """
    filename, folder_path, csv_data = image_info
    image_path = os.path.join(folder_path, filename)
    unique_id = os.path.splitext(filename)[0]
    context = csv_data.get(unique_id, "No context available")
    logging.info(f"Processing {filename}...")
    annotation = annotate_image(image_path, context)
    if annotation:
        return {
            'filename': filename,
            'unique_id': unique_id,
            'context': context,
            'annotation': annotation
        }
    else:
        logging.warning(f"Failed to annotate {filename}")
        return None

def process_images_in_folder(folder_path, csv_data, output_path):
    """
    Process all images in the specified folder, annotating them using the CSV data.
    """
    images = [
        (filename, folder_path, csv_data)
        for filename in sorted(os.listdir(folder_path))
        if any(filename.lower().endswith(ext) for ext in image_extensions)
    ]

    for image_info in tqdm(images, desc="Processing images", unit="image"):
        result = process_image(image_info)
        if result:
            # Write the result immediately after processing
            write_results([result], output_path)

def main():
    parser = argparse.ArgumentParser(description="Annotate images using the Phi-3-vision-128k-instruct model.")
    parser.add_argument('--data_dir', type=str, default="consolidated_data",
                        help="Path to the directory containing images and CSV file.")
    parser.add_argument('--output_annotation_file', type=str, default="image_annotations.csv",
                        help="Path to the output CSV file for annotations.")
    args = parser.parse_args()
    global DATA_DIR
    DATA_DIR = args.data_dir

    folder_path = os.path.join(DATA_DIR, "images")
    csv_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    output_path = args.output_annotation_file

    if not os.path.exists(folder_path):
        logging.error(f"Image folder not found: {folder_path}")
        return

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return

    csv_data = read_csv_data(csv_path)
    if not csv_data:
        return

    if not os.path.exists(output_path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'unique_id', 'context', 'description', 'bias_analysis', 'bias_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    process_images_in_folder(folder_path, csv_data, output_path)

if __name__ == "__main__":
    main()
