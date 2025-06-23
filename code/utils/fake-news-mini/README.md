# 📰 Fact-Checking Pipeline

A modular pipeline for fact-checking and fake news detection using LLMs like LLaMA, Mistral, and Phi-3. It includes summarization, multi-model inference, majority voting, and metrics evaluation.

---

### Modules

| File                  | Description                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `parallel_summary.py` | Summarizes long articles using BART.                              |
| `llama3_5shot.py`     | Labels summarized articles using LLaMA 3.1 with 5-shot prompting. |
| `summary_labels.py`   | Labels with Mistral-7B.                                           |
| `Phi_3_inference.py`  | Labels with Phi-3.                                                |
| `majority_vote.py`    | Computes majority label across models.                            |
| `metrics_calc.py`     | Computes accuracy, precision, recall, F1 using ground truth.      |
| `train_test_val.py`   | Splits dataset into train, val, and test sets.                    |

---

### Dependencies

* Python ≥ 3.8
* Transformers
* OpenAI SDK
* Scikit-learn
* Pandas
* Torch
* tqdm

---