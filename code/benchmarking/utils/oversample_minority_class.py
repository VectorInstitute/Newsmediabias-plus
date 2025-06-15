import logging
import os
import random
import argparse

import nlpaug.augmenter.word as naw
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)

AUG_BATCH_SIZE = 64
en_de_back_translator = naw.BackTranslationAug(
    from_model_name="Helsinki-NLP/opus-mt-en-de",
    to_model_name="Helsinki-NLP/opus-mt-mul-en",
    max_length=512,
    batch_size=AUG_BATCH_SIZE,
    verbose=1,
    device="cuda",
)
en_es_back_translator = naw.BackTranslationAug(
    from_model_name="Helsinki-NLP/opus-mt-en-es",
    to_model_name="Helsinki-NLP/opus-mt-mul-en",
    max_length=512,
    batch_size=AUG_BATCH_SIZE,
    verbose=1,
    device="cuda",
)
en_ru_back_translator = naw.BackTranslationAug(
    from_model_name="Helsinki-NLP/opus-mt-en-ru",
    to_model_name="Helsinki-NLP/opus-mt-mul-en",
    max_length=512,
    batch_size=AUG_BATCH_SIZE,
    verbose=1,
    device="cuda",
)
contextual_word_embs = naw.ContextualWordEmbsAug(
    model_path="roberta-base",
    action="substitute",
    top_k=50,
    aug_min=5,
    aug_max=None,
    aug_p=0.1,
    batch_size=AUG_BATCH_SIZE,
    verbose=1,
    device="cuda",
)


def _flatten_list(nested_list):
    return [  # flatten nested list
        item
        for sublist in nested_list
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]


def oversample_minority_class_via_augmentation(
    examples: dict[str, list],
) -> dict[str, list]:
    minority_indices = []
    articles = []
    for i in range(len(examples["combined_label"])):
        if (
            examples["combined_label"][i] == "Unbiased"
            or examples["text_label"][i] == "Unbiased"
        ):
            minority_indices.append(i)
            articles.append(examples["text_content"][i])

    if len(minority_indices) == 0:
        return examples

    if random.random() < 0.5:
        num_repeat_ctx_word_embs = 3
        articles = [
            contextual_word_embs.augment(articles)
            for _ in range(num_repeat_ctx_word_embs)
        ]
        articles = _flatten_list(articles)
        minority_indices = minority_indices * num_repeat_ctx_word_embs  # repeat indices

    articles = [
        en_de_back_translator.augment(articles),
        en_es_back_translator.augment(articles),
        en_ru_back_translator.augment(articles),
    ]
    articles = _flatten_list(articles)
    minority_indices = minority_indices * 3  # repeat indices

    # add new articles to 'example' but repeat the values of all other columns
    examples["text_content"].extend(articles)
    for key in examples:
        if key != "text_content":
            examples[key].extend([examples[key][i] for i in minority_indices])

    return examples


def main(args):
    ds = load_dataset(
        "csv",
        data_files=os.path.join(
            args.data_root_dir, 
            "annotations", 
            "train_topic_filtered.csv"
        ),
        split="train",
    )
    ds = ds.map(
        oversample_minority_class_via_augmentation,
        batched=True,
        batch_size=128,
        desc="Augmenting",
    )

    # save augmented dataset
    ds.to_csv(
        os.path.join(
            args.data_root_dir, 
            "annotations", 
            "train_augmented_topic_filtered.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", 
                        type=str, 
                        required=True)
    args = parser.parse_args()
    main(args)
