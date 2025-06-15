import openai
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset, Features, Value, Image as HFImage
from PIL import Image
import io
import base64
from torch.utils.data import DataLoader


# Define dataset features
features = Features({
    "unique_id": Value("string"),
    "image": HFImage(),  # Assume images are stored in the dataset
})

# Encode a PIL image to Base64
def encode_image_to_base64(image):
    """
    Convert a PIL image to a Base64 encoded string.
    Args:
        image (PIL.Image): The image to encode.
    Returns:
        str: Base64 encoded string of the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # Save the PIL image as JPEG
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"

# Load dataset
def load_hf_dataset(dataset_name, split):
    """
    Load a dataset from Hugging Face.
    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset (e.g., 'train', 'test').
    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, 
                           split=split)
    print(f"Dataset size: {len(dataset)} entries")
    return dataset

# Collate function for DataLoader
def collate_fn(batch):
    """
    Custom collate function to handle batches of images and IDs.
    Args:
        batch (list): A list of dictionaries containing 'image' and 'unique_id'.
    Returns:
        dict: A dictionary with 'images' and 'ids'.
    """
    return {
        'images': [item['image'] for item in batch],
        'ids': [item['unique_id'] for item in batch]
    }

# Generate attributes using GPT-4
def generate_attributes_openai(dataloader, openai_api_key, save_path):
    """
    Generate attributes for images using OpenAI GPT-4.
    Args:
        dataloader (DataLoader): DataLoader containing images and IDs.
        openai_api_key (str): OpenAI API key for authentication.
        save_path (str): Path to save the generated attributes.
    Returns:
        list: A list of dictionaries containing image IDs and their attributes.
    """
    openai.api_key = openai_api_key
    results = []
    with tqdm(total=len(dataloader), unit="batch") as pbar:
        for batch in dataloader:
            images, ids = batch['images'], batch['ids']
            batch_results = []

            for img, unique_id in zip(images, ids):
                try:
                    # Convert PIL image to Base64
                    img_base64 = encode_image_to_base64(img)

                    # Attributes generation prompt
                    prompt_attributes = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Look at the image and identify whether it shows any of the following concepts: Gender, Age, Ethnicity, Sport, Occupation. List only the concepts you can see, separated by commas. If none of these concepts are visible, respond with 'None' and nothing else."},
                            {"type": "image_url", 
                             "image_url": {"url": img_base64}}
                        ]
                    }

                    response_attributes = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[prompt_attributes],
                        max_tokens=50,
                        temperature=0.7
                    )
                    attributes = response_attributes['choices'][0]['message']['content'].strip()

                    batch_results.append({
                        "id": unique_id,
                        "attributes": attributes,
                    })

                except Exception as e:
                    print(f"Error generating attributes for ID {unique_id}: {e}")

            results.extend(batch_results)

            # Save results incrementally after every batch
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

            pbar.update(1)

    return results

# Argument parsing
def parse_args():
    parser = ArgumentParser(description="Generate attributes for images using OpenAI GPT-4.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset")
    parser.add_argument("--split", type=str, required=True, help="The split of the dataset (e.g., 'train', 'test')")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for processing images")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated attributes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load dataset
    dataset = load_hf_dataset(args.dataset_name, 
                              args.split)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    # Generate attributes
    generate_attributes_openai(dataloader, 
                               args.openai_api_key, 
                               args.save_path)


# To run the Python script:
# python generate_attributes.py \
#     --dataset_name vector-institute/newsmediabias-plus-clean \
#     --split train \
#     --batch_size 32 \
#     --openai_api_key "OPENAI_API_KEY" \
#     --save_path attributes_full.json
