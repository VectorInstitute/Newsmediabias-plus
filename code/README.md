# Codebase

This directory contains all major components for processing, analyzing, evaluating, and benchmarking media bias, disinformation, and demographics in news articles.

---

## Folder Structure

| Folder          | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `Annotators/`   | Manual and automated annotation logic for labeling bias/disinfo attributes.  |
| `benchmarking/` | Scripts to compare SLMs vs LLMs for bias/disinformation classification.      |
| `demographics/` | Extracts demographic signals (e.g., gender, race, ideology) from news text.  |
| `Evaluations/`  | Evaluation scripts and configs for multimodal LLMs and classification tasks. |
| `NER/`          | Named Entity Recognition modules tailored to news data.                      |
| `projects/`     | Specialized sub-projects or experiments related to bias/disinfo.             |
| `scrapper/`     | Web scraping tools for collecting and structuring news articles.             |
| `utils/`        | Shared helper scripts (e.g., dataset splitting, augmentation).               |
| `VQA/`          | Visual Question Answering on news images using multimodal models.            |