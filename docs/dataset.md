# Dataset Details

## Overview

Our projects maintain a collection of datasets hosted on Hugging Face, focusing on human-centered AI, multimodal disinformation detection, and news media bias analysis. These datasets support reproducible research and benchmark development for responsible AI.

---

## Hugging Face Datasets & Repositories

### [HumaniBench](https://huggingface.co/datasets/vector-institute/HumaniBench)
- **Description:** A large-scale benchmark with 32,000+ human-verified multilingual image-question pairs for evaluating fairness, ethics, reasoning, and empathy in Large Multimodal Models.
- **Stats:** 32.6k samples, 10+ languages, visual and textual annotations.
- **Link:** [HumaniBench Dataset](https://huggingface.co/datasets/vector-institute/HumaniBench)

### [VLDBench](https://huggingface.co/datasets/vector-institute/VLDBench)
- **Description:** Benchmark for multimodal disinformation detection with 62,000+ real-world image-article pairs, human verified by domain experts.
- **Stats:** 62k+ samples, 58 news sources, multimodal labels.
- **Link:** [VLDBench Dataset](https://huggingface.co/datasets/vector-institute/VLDBench)

### [NewsMediaBias-Plus](https://huggingface.co/datasets/vector-institute/newsmediabias-plus)
- **Description:** Comprehensive news articles dataset spanning multiple ideological sources, annotated for bias in text and images.
- **Stats:** 40.9k+ samples, multi-outlet, includes bias annotations and metadata.
- **Link:** [NewsMediaBias-Plus Dataset](https://huggingface.co/datasets/vector-institute/newsmediabias-plus)

### [NMB-Plus Bias NER BERT](https://huggingface.co/vector-institute/nmb-plus-bias-ner-bert)
- **Description:** Named Entity Recognition model fine-tuned on NewsMediaBias-Plus for bias detection in entity mentions.
- **Link:** [NER Model](https://huggingface.co/vector-institute/nmb-plus-bias-ner-bert)

### [Llama3.2 Multimodal Newsmedia Bias Detector](https://huggingface.co/vector-institute/Llama3.2-Multimodal-Newsmedia-Bias-Detector)
- **Description:** Multimodal bias detection model leveraging Llama3.2 architecture to identify bias in combined text and images.
- **Link:** [Multimodal Bias Detector](https://huggingface.co/vector-institute/Llama3.2-Multimodal-Newsmedia-Bias-Detector)

### [Llama3.2 NLP Newsmedia Bias Detector](https://huggingface.co/vector-institute/Llama3.2-NLP-Newsmedia-Bias-Detector)
- **Description:** NLP-based bias detector using Llama3.2, specialized for textual bias analysis in news media.
- **Link:** [NLP Bias Detector](https://huggingface.co/vector-institute/Llama3.2-NLP-Newsmedia-Bias-Detector)

### Additional Repositories and Models
- [NMB-Plus Clean Dataset](https://huggingface.co/vector-institute/nmb-plus-clean) — Cleaned news media bias dataset (31.3k samples).
- [maximuspowers/nmbp-bert-full-articles](https://huggingface.co/maximuspowers/nmbp-bert-full-articles) — BERT-based text classification on full articles.
- [maximuspowers/multimodal-bias-classifier](https://huggingface.co/maximuspowers/multimodal-bias-classifier) — Multimodal bias classifier.
<!-- - [NMB-Plus Named Entities](https://huggingface.co/vector-institute/NMB-Plus-Named-Entities) — Named entity viewer dataset for bias assessment. -->

---

## Dataset Access & Usage

All datasets can be loaded via the Hugging Face `datasets` library. Example:

```python
from datasets import load_dataset

# HumaniBench datasets by task
scene_understanding_ds = load_dataset("vector-institute/HumaniBench", "task1_Scene_Understanding")
instance_identity_ds = load_dataset("vector-institute/HumaniBench", "task2_Instance_Identity")
multiple_choice_vqa_ds = load_dataset("vector-institute/HumaniBench", "task3_Multiple_Choice_VQA")
multilingual_open_ended_ds = load_dataset("vector-institute/HumaniBench", "task4_Multilingual_OpenEnded")
multilingual_close_ended_ds = load_dataset("vector-institute/HumaniBench", "task4_Multilingual_CloseEnded")
visual_grounding_ds = load_dataset("vector-institute/HumaniBench", "task5_Visual_Grounding")
empathetic_captioning_ds = load_dataset("vector-institute/HumaniBench", "task6_Empathetic_Captioning")
image_resilience_ds = load_dataset("vector-institute/HumaniBench", "task7_Image_Resilience")

# Other datasets
vldbench_ds = load_dataset("vector-institute/VLDBench")
newsmediabias_plus_ds = load_dataset("vector-institute/newsmediabias-plus")
nmb_plus_clean_ds = load_dataset("vector-institute/nmb-plus-clean")
nmb_plus_named_entities_ds = load_dataset("vector-institute/NMB-Plus-Named-Entities")

```


---

## News Sources & Coverage

Our datasets cover a wide spectrum of news sources, including major US outlets, global media, and diverse political perspectives, ensuring comprehensive bias analysis capabilities.

Refer to the detailed news sources list in the section below:

- Major U.S. News Outlets: CNN, Fox News, CBS News, ABC News, New York Times, Washington Post, USA Today, Wall Street Journal, AP News, Politico, New York Post, Forbes, Reuters, Bloomberg
- Global & Alternative News Sources: BBC, Al Jazeera, PBS NewsHour, The Guardian, Newsmax, HuffPost, CNBC, C-SPAN, The Economist, Financial Times, Time, Newsweek, The Atlantic, The New Yorker, The Hill, ProPublica, Axios
- Conservative & Progressive News Outlets: National Review, The Daily Beast, Daily Kos, Washington Examiner, The Federalist, OANN, Daily Caller, Breitbart
- Canadian News Sources: CBC, Toronto Sun, Global News, The Globe and Mail, National Post

<!-- Keep your existing news sources, schema, sample data sections here -->


---
