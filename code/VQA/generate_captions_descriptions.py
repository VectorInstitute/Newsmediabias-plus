import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset, Features, Value, Image as HFImage
from tqdm import tqdm
import json
from argparse import ArgumentParser
from huggingface_hub import login

# Define dataset features
features = Features({
    "unique_id": Value("string"),
    "image": HFImage(),
})

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

# Caption generation function
def generate_captions(model, dataloader, processor, device, save_path):
    """
    Generate captions and descriptions for images in the dataset.
    Args:
        model (AutoModelForVision2Seq): The pre-trained model for caption generation.
        dataloader (DataLoader): DataLoader for the dataset.
        processor (AutoProcessor): Processor to handle image and text inputs.
        device (torch.device): Device to run the model on.
        save_path (str): Path to save the generated captions and descriptions.
    Returns:
        list: List of generated captions and descriptions.
    """
    model.eval()
    results = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            for batch in dataloader:
                images, ids = batch['images'], batch['ids']
                batch_results = []
                for img, id in zip(images, ids):
                    try:
                        # Generate caption
                        prompt = [
                            {"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", 
                                 "text": "Provide a descriptive one-sentence caption for the given image: "}
                            ]}
                        ]

                        input_text = processor.apply_chat_template(
                            prompt,
                            add_generation_prompt=True
                        )
                        inputs = processor(
                            img, 
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(device)

                        output = model.generate(**inputs, 
                                                max_new_tokens=50, 
                                                temperature=0.7, 
                                                top_p=0.9)
                        caption = processor.decode(output[0], 
                                                   skip_special_tokens=True)

                        # Generate image description
                        prompt = [
                            {"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", 
                                 "text": "Provide a one-paragraph description for the given image: "}
                            ]}
                        ]

                        input_text = processor.apply_chat_template(
                            prompt,
                            add_generation_prompt=True
                        )
                        inputs = processor(
                            img, 
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(device)

                        output = model.generate(**inputs, 
                                                max_new_tokens=150, 
                                                temperature=0.7, 
                                                top_p=0.9)
                        img_description = processor.decode(output[0], 
                                                           skip_special_tokens=True)

                        batch_results.append({
                            "id": id,
                            "caption": caption,
                            "img_description": img_description
                        })

                    except Exception as e:
                        print(f"Error generating caption for ID {id}: {e}")

                results.extend(batch_results)

                # Save results incrementally after every batch
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)

                pbar.update(1)
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated captions")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace authentication token")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")

    args = parser.parse_args()

    # Login to HuggingFace
    login(token=args.hf_token)

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = load_hf_dataset(args.dataset_name, 
                              args.split)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    # Load model and processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Generate captions
    generate_captions(model, 
                      dataloader, 
                      processor, 
                      device, 
                      args.save_path)


# To run the Python script:
# python generate_captions_descriptions.py \
#     --dataset_name vector-institute/newsmediabias-plus-clean \
#     --split train \
#     --batch_size 64 \
#     --save_path captions_descriptions.json \
#     --hf_token "YOUR_HF_TOKEN" \
#     --model_path /model-weights/Llama-3.2-11B-Vision-Instruct