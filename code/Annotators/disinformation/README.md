# Disinformation Detection

This module provides tools for detecting **disinformation** and identifying **rhetorical techniques** in news text using models like `GPT-4o`, `Mistral-7B`, and `Mistral-Large`.

---

## Contents

| Script                                  | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `detect_disinfo_binary.py`              | Binary classification (`Likely` / `Unlikely`) using OpenAI GPT-4o. |
| `label_rhetoric_disinfo_full.py`        | Multi-label rhetorical technique detection + disinfo using GPT-4o. |
| `mistral_disinfo_labeler_binary.py`     | Binary classification using self-hosted Mistral-7B.                |
| `mistral_disinfo_labeler_multilabel.py` | Multi-label rhetorical detection + disinfo using Mistral-Large.    |

---

## Features

* 🔍 Supports both **binary** and **multi-label** disinformation analysis
* ⚡ Asynchronous batch processing for Mistral-based pipelines
* 💾 Checkpointing for fault tolerance and resumability
* 🧠 Prompts include detailed rhetorical technique criteria
* 📄 Outputs include structured CSVs with prediction and reasoning

---

## Output

Each script saves results to a CSV file, with columns such as:

* `Disinformation`: Likely / Unlikely
* `Reasoning`: Short explanation (if applicable)
* Optional: One column per rhetorical technique (Present / Absent)