import json
import os
import signal
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct"

PROMPT_TEMPLATE = """
<|user|>\n<|image_1|>
Analyze the given image and the following context, if present, from a news article and perform the following tasks.
Respond with a python dictionary with the keys specified for each task.

Context:
{context}

Tasks:
1. Describe the image in detail including its content, including any relevant objects, people, actions, or events depicted.
key: description
2. Analyze the image and the provided context for potential biases and summarize your analysis in 2 or 3 paragraphs, referencing specific elements of the image and/or context.
Consider the following, and any other relevant factors, in your analysis (mention the key factors you used in the analysis):
- Why might this particular image have been chosen? Would another image change the message/tone significantly?
- Consider how people, places or things are depicted. Are they shown in a positive, negative or neutral light?
- Is the image likely to evoke strong emotions in viewers that may sway its viewers? If so, what kind?
- Does the image reinforce stereotypes about groups of people or attempt to oversimplify a complex issue?
- Look at how the image is cropped or framed - what's included vs. excluded? Note the angle, perspective and composition.
- Look for signs the image may have been digitally altered. Consider if the scene appears artificially staged rather than candid.
- Consider how the headline frames the image and how the image might influence the reader's interpretation of the headline.
key: bias_analysis
3. Classify the image as one of the following categories: Biased, Unbiased.
key: label

Do not return anything except the python dictionary of key-value pairs as output.
<|end|>\n<|assistant|>\n
"""

num_samples = 15


def save_annotation(annotations: list[dict], annotation_file: str):
    """
    Save the annotations to a CSV file.
    """
    results_df = pd.DataFrame(annotations)
    results_df.to_csv(annotation_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Annotation Script")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="consolidated_data",
        help="Directory containing the data files.",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="phi3_vision_128k_annotations.csv",
        help="File to save the annotations.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    annotations = []

    def signal_handler(sig, frame):
        save_annotation(annotations, args.annotation_file)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # handle ctrl+c
    signal.signal(signal.SIGTERM, signal_handler)  # handle slurm exit signal
    signal.signal(signal.SIGUSR1, signal_handler)  # handle slurm exit signal

    meta_df = pd.read_csv(os.path.join(data_dir, "cleaned_data.csv"))

    for row in tqdm(
        meta_df.itertuples(index=False, name="Row"),
        total=len(meta_df),
        desc="Annotating",
    ):
        uid = getattr(row, "unique_id")

        # image extension could be jpg, jpeg, or png
        image_path = os.path.join(data_dir, "images", f"{uid}")
        image_extensions = [".jpg", ".jpeg", ".png"]
        for extension in image_extensions:
            if os.path.exists(image_path + extension):
                image_path += extension
                break

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        context = """
        Headline: {headline}
        Date: {date}
        """

        prompt = PROMPT_TEMPLATE.format(
            context=context.format(
                headline=getattr(row, "title"), date=getattr(row, "date_published")
            ),
        )

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1536,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        annotation = {
            "unique_id": uid,
            "image_path": image_path,
        }
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            response_dict = {
                "description": None,
                "bias_analysis": None,
            }

        annotation.update(response_dict)
        annotations.append(annotation)

    save_annotation(annotations, args.annotation_file)
