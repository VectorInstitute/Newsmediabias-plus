# Image Captioning and Attribute Extraction

This folder contains two Python scripts to process images from a Hugging Face dataset:

### 1. `generate_captions_descriptions.py`

Generates:

* **One-sentence captions**
* **One-paragraph descriptions**

using a Hugging Face Vision-to-Text model (e.g., LLaVA, LLaMA variants).


---

### 2. `generate_attributes.py`

Extracts visible **concepts** from images (e.g., Gender, Age, Ethnicity, Sport, Occupation) using **OpenAI GPT-4o**.

---

### Requirements

* `torch`, `transformers`, `datasets`, `Pillow`, `openai`, `tqdm`
* Hugging Face and/or OpenAI API access

---

Let me know if you want an extended version with setup instructions or environment requirements.
