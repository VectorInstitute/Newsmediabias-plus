# HumaniBench: A Human-Centric Benchmark for Large Multimodal Models Evaluation

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebed8e26-5bdf-48c1-ae41-0775b8c33c0a" alt="HumaniBench Logo" width="300"/>
</p>

<p align="center">
  🌐 <b><a href="https://vectorinstitute.github.io/HumaniBench/" target="_blank" rel="noopener noreferrer">Website</a></b> &nbsp;|&nbsp;
  📄 <b><a href="https://arxiv.org/abs/2505.11454" target="_blank" rel="noopener noreferrer">Paper</a></b> &nbsp;|&nbsp;
  📊 <b><a href="https://huggingface.co/datasets/vector-institute/HumaniBench" target="_blank" rel="noopener noreferrer">Dataset</a></b>
</p>



---

---

## 🧠 Overview

As multimodal generative AI systems become increasingly integrated into human-centered applications, evaluating their **alignment with human values** has become critical.

**HumaniBench** is the **first comprehensive benchmark** designed to evaluate **Large Multimodal Models (LMMs)** on **seven Human-Centered AI (HCAI) principles**:

* **Fairness**
* **Ethics**
* **Understanding**
* **Reasoning**
* **Language Inclusivity**
* **Empathy**
* **Robustness**

![VLDBench Pipeline](images/HumaniBenchTeaser.jpg)


---

## 📦 Features

* 📷 **32,000+ Real-World Image–Question Pairs**
* ✅ **Human-Verified Ground Truth Annotations**
* 🌐 **Multilingual QA Support (10+ languages)**
* 🧠 **Open and Closed-Ended VQA Formats**
* 🧪 **Visual Robustness & Bias Stress Testing**
* 📑 **Chain-of-Thought Reasoning + Perceptual Grounding**


---

## 📂 Evaluation Tasks Overview

| Task                               | Focus                                                                                          | 
| :--------------------------------- | :--------------------------------------------------------------------------------------------- |
| **Task 1: Scene Understanding**    | Visual reasoning + bias/toxicity analysis in social attributes (gender, age, occupation, etc.) |
| **Task 2: Instance Identity**      | Visual reasoning in culturally rich, socially grounded settings                                |
| **Task 3: Multiple Choice QA**     | Structured attribute recognition via multi-choice questions                                    |
| **Task 4: Multilingual Visual QA** | VQA across 10+ languages, including low-resource ones                                          |
| **Task 5: Visual Grounding**       | Bounding box localization of socially salient regions                                          |
| **Task 6: Empathetic Captioning**  | Human-style emotional captioning evaluation                                                    |
| **Task 7: Image Resilience**       | Robustness testing via image perturbations                                                     |

---

## 🧬 Pipeline

**Three-stage process:**

1. **Data Collection:** Curated from global news imagery, tagged by social attributes (age, gender, race, occupation, sport)

2. **Annotation:** GPT-4o–assisted labeling + human expert verification

3. **Evaluation:** Comprehensive scoring across Accuracy, Fairness, Robustness, Empathy and Faithfulness


---

## 🔑 Key Insights

* 🔍 **Bias persists**, especially across gender and race
* 🌐 **Multilingual gaps** affect low-resource language performance
* ❤️ **Empathy and ethics** vary significantly by model family
* 🧠 **Chain-of-Thought reasoning** improves performance but doesn’t fully mitigate bias
* 🧪 **Robustness tests** reveal fragility to noise, occlusion, and blur

---

## 📚 Citation

If you use HumaniBench or this evaluation suite in your work, please cite:

```bibtex
@misc{raza2025humanibenchhumancentricframeworklarge,
        title={HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation}, 
        author={Shaina Raza and Aravind Narayanan and Vahid Reza Khazaie and Ashmal Vayani and Mukund S. Chettiar and Amandeep Singh and Mubarak Shah and Deval Pandya},
        year={2025},
        eprint={2505.11454},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2505.11454}, 
  }

```

---

## 📬 Contact

For questions, collaborations, or dataset access requests, please contact the corresponding author at [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai).

---

### ⚡ HumaniBench promotes trustworthy, fair, and human-centered multimodal AI.

**We invite researchers, developers, and policymakers to explore, evaluate, and extend HumaniBench. 🚀**

---
