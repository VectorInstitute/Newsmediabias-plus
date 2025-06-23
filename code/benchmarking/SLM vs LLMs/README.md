# 📊 Benchmarking: SLMs vs LLMs

This module evaluates and compares traditional **Small Language Models (SLMs)** and state-of-the-art **Large Language Models (LLMs)** on disinformation classification tasks.

---

## Directory Structure

```
Benchmarking/
    ├── LLM/
    ├── SLM/
    ├── utils.py
    ├── augmentation.py
```

---

## Models

### LLM Folder

Includes three evaluation settings per LLM:

* `*_ft.py` — Fine-tuned for sequence classification.
* `*_ift.py` — Instruction fine-tuned (causal generation with prompts).
* `*_shot.py` — Few-shot inference (0-shot or 2-shot) using prompting.

Supported LLMs:

* Gemma (`gemma_ft.py`, `gemma_ift.py`, `gemma_shot.py`)
* LLaMA-3.1 & 3.2
* Mistral

Each script:

* Performs 5-fold training + early stopping
* Tracks emissions (`codecarbon`) & logs to `wandb`
* Supports both prediction and evaluation (accuracy, precision, recall, F1)

---

### SLM Folder

Contains fine-tuned classifiers:

* BERT variants (`bert_base.py`, `bert_large.py`)
* RoBERTa (`roberta.py`)
* DistilBERT (`distilbert.py`)
* GPT-2 (`gpt2.py`)
* BART (`bart.py`)

Features:

* Token classification via Hugging Face Transformers
* 5-fold cross-validation
* Support for class balancing and loss customization (CrossEntropy, FocalLoss)
* Performance logging and plots

---

## Utilities

| File              | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `utils.py`        | Splits datasets, creates balanced train/test subsets                             |
| `augmentation.py` | Data augmentation via **backtranslation (EN→FR→EN)** and **synonym replacement** |

---

## Evaluation Metrics

* Accuracy, Precision, Recall, F1 (weighted)
* Confusion matrix
* Per-class classification report
* Carbon emissions via `codecarbon`
* Learning curves & heatmaps (auto-saved)