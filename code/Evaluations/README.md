# 🤖 LMM Evaluation Codes

This folder contains **evaluation scripts** for benchmarking Large Multimodal Models (LMMs) on the [NewsMediaBias+](https://huggingface.co/datasets/vector-institute/newsmediabias-plus-clean) dataset. Each script evaluates a specific model's ability to detect **disinformation** using both article text and images.

---

### Files Overview

| File                                                               | Purpose                                                                                      |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `deepseek_*.py`, `InternVL.py`, `llava*.py`, `llama3.2-*.py`, etc. | Model-specific scripts to perform image-text evaluation on articles.                         |
| `judge-nlp.py`, `judge-vision.py`                                  | Baseline judgment models (text-only and vision-enhanced).                                    |
| `eval.py`                                                          | Calculates **accuracy**, **precision**, **recall**, and **F1-score** from JSON result files. |
| `results.json` *(generated)*                                       | Output file with predictions, ground truth, and prompt used per sample.                      |

---

### Key Features

* Shared disinformation checklist prompt across models
* Supports **Likely / Unlikely** binary classification
* Modular design for extending to new LMMs
* Outputs per-sample JSON and aggregate metrics
