# Benchmarking Annotation Framework

## Overview

This page presents the performance benchmarking of Small Language Models (SLMs) and Large Language Models (LLMs) within our annotation framework. The objective is to evaluate how these models perform in tasks involving text and multimodal data (text + image). For this benchmarking, **SLMs** are defined as models with fewer parameters, typically below 15 million, such as BERT and GPT-2. In contrast, **LLMs**—including models like Llama3, Mistral, Gemma, and Phi—possess hundreds of millions to billions of parameters. This scale difference highlights the trade-off between efficiency and complexity when handling various tasks and datasets.

## Benchmarking Results: Text-Based Models

The following table summarizes the performance of state-of-the-art text-based models. "FT" stands for Fine Tuning, "IFT" for Instruction Fine Tuning, and models have been evaluated across different metrics such as Precision, Recall, F1-Score, and Accuracy.

| Model                              | Architecture             | Precision | Recall | F1-Score | Accuracy |
| ---------------------------------- | ------------------------ | --------- | ------ | -------- | -------- |
| **BERT-base-uncased (FT)**         | Encoder-only             | 0.8887    | 0.887  | 0.8846   | 0.8870   |
| **DistilBERT (FT)**                | Encoder-only             | 0.8665    | 0.8554 | 0.8487   | 0.8661   |
| **RoBERTa-base (FT)**              | Encoder-only             | 0.894     | 0.894  | 0.8927   | 0.8940   |
| **GPT2 (FT)**                      | Decoder-only             | 0.8762    | 0.8751 | 0.8727   | 0.8751   |
| **BART (FT)**                      | Encoder-decoder          | 0.8762    | 0.876  | 0.874    | 0.8760   |
| **Llama 3.1-8B-instruct (0-shot)** | Decoder-only             | 0.828     | 0.689  | 0.7525   | 0.7200   |
| **Llama 3.1-8B-instruct (5-shots)**| Decoder-only             | 0.840     | 0.770  | 0.8034   | 0.7905   |
| **Llama 3.1-8B-instruct (IFT)**    | Decoder-only             | 0.8019    | 0.8019 | 0.8418   | 0.8180   |
| **Llama 3.1-8B (FT)**              | Decoder-only             | 0.800     | 0.800  | 0.790    | 0.8320   |
| **Llama 3.2-3B-instruct (0-shot)** | Decoder-only             | 0.7386    | 0.4622 | 0.5012   | 0.3897   |
| **Llama 3.2-3B-instruct (5-shots)**| Decoder-only             | 0.7989    | 0.3840 | 0.3763   | 0.4622   |
| **Llama 3.2-3B-instruct (IFT)**    | Decoder-only             | 0.839     | 0.7984 | 0.8182   | 0.8084   |
| **Llama 3.2-3B-sequence classifier (FT)** | Decoder-only     | 0.840     | 0.850  | 0.840    | 0.8200   |
| **Mistral-v0.3-instruct (0-shot)** | Decoder-only             | 0.8153    | 0.5250 | 0.6380   | 0.6990   |
| **Mistral-v0.3-instruct (5-shots)**| Decoder-only             | 0.8319    | 0.8134 | 0.8225   | 0.8230   |
| **Mistral-v0.3-instruct (IFT)**    | Decoder-only             | 0.889     | 0.924  | 0.9062   | 0.8680   |
| **Mistral-v0.3 (FT)**              | Decoder-only             | 0.820     | 0.820  | 0.820    | 0.8050   |
| **qwen2.5-7B (0-shot)**            | Decoder-only             | 0.8576    | 0.8576 | 0.8576   | 0.8576   |
| **qwen2.5-7B (5-shots)**           | Decoder-only             | 0.8660    | 0.8790 | 0.8720   | 0.8900   |

## Benchmarking Results: Multimodal Models (Text + Image)

The following table highlights the performance of multimodal models, which process both text and image inputs. These models were evaluated on weighted averages for Precision, Recall, and F1.

| Model                                    | FT/IFT   | Architecture (Text-Image)         | Precision   | Recall      | F1          |
| ---------------------------------------- | -------- | --------------------------------- | ----------- | ----------- | ----------- |
| **SpotFake (XLNET + VGG-19)**            | FT       | Encoder-Encoder                   | 0.7415      | 0.6790      | 0.7040      |
| **BERT + ResNet-34**                     | FT       | Encoder-Encoder                   | 0.8311      | 0.6277      | 0.6745      |
| **FND-CLIP (BERT and CLIP/ResNET + CLIP)** | FT      | Encoder-Encoder                   | 0.6935      | 0.7151      | 0.7035      |
| **InstructBlipV (Text + Images)**        | FT       | Encoder-Encoder                   | -           | -           | -           |
| **Distill-bert/CLIP**                    | FT       | Encoder-Encoder                   | 0.5472      | 0.4047      | 0.4652      |
| **google/paligemma-3b-pt-224**           | IFT      | Decoder-Encoder                   | -           | -           | -           |
| **microsoft/Phi-3-vision-128k-instruct** | 0-shot   | Decoder-Encoder                   | -           | -           | -           |
| **microsoft/Phi-3-vision-128k-instruct** | 5-shots  | Decoder-Encoder                   | -           | -           | -           |
| **Pixtral-12B-2409**                     | 0-shot   | Decoder-Encoder                   | -           | -           | -           |
| **Pixtral-12B-2409**                     | 2-shots  | Decoder-Encoder                   | -           | -           | -           |
| **LLaVA-1.6**                            | 0-shot   | Decoder-Encoder                   | -           | -           | -           |
| **Llama-3.2-11B-Vision-Instruct**        | 0-shot   | Decoder-Encoder                   | -           | -           | -           |
| **meta-llama/Llama-Guard-3-11B-Vision**  | FT       | Decoder-Encoder                   | -           | -           | -           |

---

This benchmarking offers an insightful overview of how various models, ranging from smaller to large-scale, perform in distinct environments and tasks. The text-based and multimodal benchmarks reflect the strength of these models in handling the complexities of both textual data and combined text-image inputs, providing a useful reference for selecting the appropriate model based on the task requirements.
