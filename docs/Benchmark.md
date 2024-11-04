# Benchmarking Annotation Framework

## Overview

This page presents the performance benchmarking of Small Language Models (SLMs) and Large Language Models (LLMs) within our annotation framework. The objective is to evaluate how these models perform in tasks involving text and multimodal data (text + image). For this benchmarking, **SLMs** are defined as models with fewer parameters, typically below 15 million, such as BERT and GPT-2. In contrast, **LLMs**—including models like Llama3, Mistral, Gemma, and Phi—possess hundreds of millions to billions of parameters. This scale difference highlights the trade-off between efficiency and complexity when handling various tasks and datasets.

## Benchmarking Results: Text-Based Models

| Model | Configuration | Precision | Recall | F1 | Test Accuracy |
|--------|---------------|-----------|--------|----|---------------|
| **Small Language Models** |  |  |  |  |  |
| BERT-base-uncased | FT | 0.8887 | 0.8870 | 0.8878 | 0.8870 |
| DistilBERT | FT | 0.8665 | 0.8554 | 0.8609 | 0.8710 |
| RoBERTa-base | FT | 0.8940 | 0.8940 | 0.8940 | 0.8940 |
| GPT2 | FT | 0.8762 | 0.8751 | 0.8756 | 0.8751 |
| BART | FT | 0.8762 | 0.8760 | 0.8761 | 0.8760 |
| **Large Language Models** |  |  |  |  |  |
| Llama 3.1-8B-instruct | 0-shot | 0.8280 | 0.6890 | 0.7521 | 0.7200 |
|  | 5-shot | 0.8400 | 0.7700 | 0.8035 | 0.7905 |
|  | IFT | 0.8019 | 0.8019 | 0.8019 | 0.8180 |
| Llama 3.1-8B (base) | FT | 0.8800 | 0.8600 | 0.8699 | 0.8320 |
| Llama 3.2-3B-instruct | 0-shot | 0.7386 | 0.7550 | 0.7467 | 0.6897 |
|  | 5-shot | 0.7989 | 0.6840 | 0.7370 | 0.6133 |
|  | IFT | 0.8390 | 0.7984 | 0.8182 | 0.8084 |
| Llama 3.2-3B (base) | FT | 0.8400 | 0.8500 | 0.8450 | 0.8200 |
| Mistral-v0.3 7B-instruct | 0-shot | 0.8153 | 0.5250 | 0.6387 | 0.6990 |
|  | 5-shot | 0.8319 | 0.8134 | 0.8225 | 0.7830 |
|  | IFT | 0.8890 | 0.9240 | 0.9062 | 0.7980 |
| Mistral-v0.3 7B (base) | FT | 0.8200 | 0.7400 | 0.7779 | 0.8014 |
| Qwen2.5-7B | 0-shot | 0.8576 | 0.8576 | 0.8576 | 0.8576 |
|  | 5-shot | 0.8660 | 0.8790 | 0.8724 | 0.8900 |
|  | IFT | 0.8357 | 0.8474 | 0.8415 | 0.8474 |

**Table 1**: Performance metrics for various language models and configurations. Configuration types: 0-shot = No prior examples used for inference, 5-shot = Five examples provided for context before inference, FT = Fine-tuned on task-specific data, IFT = Instruction fine-tuned with targeted training.

| Model | Config. (Text-Image) | Precision | Recall | F1 | Test Accuracy |
|-------|-----------------------|-----------|--------|----|---------------|
| **Small Language Models** |  |  |  |  |  |
| SpotFake (XLNET + VGG-19) | FT | 0.7415 | 0.6790 | 0.7089 | 0.8151 |
| BERT + ResNet-34 | FT | 0.8311 | 0.6277 | 0.7152 | 0.8248 |
| FND-CLIP (BERT and CLIP) | FT | 0.6935 | 0.7151 | 0.7041 | 0.8971 |
| Distill-RoBERTa and CLIP | FT | 0.7000 | 0.6600 | 0.6794 | 0.8600 |
| **Large Vision Language Models** |  |  |  |  |  |
| Phi-3-vision-128k-instruct | 0-shot | 0.7400 | 0.6700 | 0.7033 | 0.7103 |
| Phi-3-vision-128k-instruct | 5-shot | 0.7600 | 0.7200 | 0.7395 | 0.7024 |
| Phi-3-vision-128k-instruct | IFT | 0.7800 | 0.8000 | 0.7899 | 0.7200 |
| LLaVA-1.6 | 0-shot | 0.7531 | 0.6466 | 0.6958 | 0.6500 |
| LLaVA-1.6 | 5-shot | 0.7102 | 0.6893 | 0.6996 | 0.6338 |
| Llama-3.2-11B-Vision-Instruct | 0-shot | 0.6668 | 0.7233 | 0.6939 | 0.7060 |
| Llama-3.2-11B-Vision-Instruct | 5-shot | 0.7570 | 0.7630 | 0.7600 | 0.7299 |
| Llama-3.2-11B-Vision-Instruct | IFT | 0.7893 | 0.8838 | 0.8060 | 0.9040 |

**Table 2**: Performance metrics for various small and large language models in text-image configurations. Configuration types: 0-shot = No prior examples used for inference, 5-shot = Five examples provided for context before inference, FT = Fine-tuning, IFT = Instruction Fine-tuning.

This benchmarking offers an insightful overview of how various models, ranging from smaller to large-scale, perform in distinct environments and tasks. The text-based and multimodal benchmarks reflect the strength of these models in handling the complexities of both textual data and combined text-image inputs, providing a useful reference for selecting the appropriate model based on the task requirements.
