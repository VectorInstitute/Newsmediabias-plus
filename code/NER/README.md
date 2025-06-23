# 🧠 Named-Entity-Recognition (NER)

A toolkit for exploring and evaluating **bias detection** in news articles using NER-style labeling powered by OpenAI LLMs. Includes dataset preparation, prompt construction, and structured annotation pipelines.

---

### Modules

| File                            | Description                                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `download_hf_dataset.py`        | Downloads and extracts the **News Media Bias+** dataset from Hugging Face into CSV format.                                             |
| `gpt-4-o_prompt_engineering.py` | Uses **GPT-4o** for token-level **NER-style bias labeling** based on prompt-engineered BIO tagging.                                    |
| `ner_annotation_module.py`      | Builds a reusable **DSPy module** to label text with **B-BIAS, I-BIAS, O** tags and aggregate multi-label annotations across examples. |

---

### Features

* Structured **bias annotation** via LLMs using custom prompt templates
* BIO tagging format for token-level **NER-style labeling**
* Supports few-shot fine-tuning via **DSPy + LabeledFewShot**
* Modular evaluation and export to CSV for downstream use