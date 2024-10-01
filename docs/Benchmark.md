# Benchmarking for Annotation Framework

## Purpose

The purpose of this benchmarking page is to evaluate the performance of Small Language Models (SLMs) and Large Language Models (LLMs) in our annotation framework. In this context, we refer to SLMs as those with fewer parameters, typically less than 15 million, such as BERT and GPT-2. LLMs, like Llama3, Mistral, Gemma, Phi, have significantly more parameters, often in the hundreds of millions to billions. This relative difference in scale allows us to compare the efficiency to handle more complex tasks and datasets, while SLMs are more efficient for simpler tasks or environments with limited resources.

## Benchmarking on Texts

### Small Language Models (SLMs)

| Model               | Training Method | Architecture            | Classes                                               | Carbon Emissions (tCO₂e) |
|---------------------|------------------|-------------------------|-------------------------------------------------------|--------------------------|
| BERT-base-uncased   | Fine-tuning      | Encoder-only            | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| BERT-large-uncased  | Fine-tuning      | Encoder-only            | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| DistilBERT          | Fine-tuning      | Encoder-only            | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| RoBERTa-base        | Fine-tuning      | Encoder-only            | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| GPT2                | Fine-tuning      | Decoder                 | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| BART                | Fine-tuning      | Encoder-decoder         | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |

### Large Language Models (LLMs)

#### Llama Models

| Model                       | Training Method               | Architecture                                 | Classes                                               | Carbon Emissions (tCO₂e) |
|-----------------------------|-------------------------------|----------------------------------------------|-------------------------------------------------------|--------------------------|
| Llama 3.1-8B-instruct        | 0-shot, 5-shot, IFT            | Decoder-only autoregressive CausalLM         | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| Llama 3.1-8B                | Fine-tuning                    | Decoder-only autoregressive sequence classification | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| Llama 3.2-1B-Instruct        | 0-shot, 5-shot, IFT            | Decoder-only autoregressive CausalLM         | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| Llama 3.2-1B                | Fine-tuning                    | Decoder-only autoregressive sequence classification | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| Llama 3.2-3B-instruct        | 0-shot, 5-shot, IFT            | Decoder-only autoregressive CausalLM         | Fake/Bias/Likely (0), Real/Unbias/Unlikely (1)        | N/A                      |
| Llama 3.2-8B-sequence classifier | Fine-tuning                  | Decoder-only autoregressive sequence classification | Fake/Bias, Real/Unbias                                | N/A                      |
| Llama 3 (70B)               | N/A                           | N/A                                          | N/A                                                   | 1900                     |

#### Other LLMs

| Model                       | Training Method               | Architecture                                 | Classes                     | Carbon Emissions (tCO₂e) |
|-----------------------------|-------------------------------|----------------------------------------------|-----------------------------|--------------------------|
| Mistral-v0.3-instruct        | 0-shot, 5-shot, IFT            | Decoder-only autoregressive CausalLM         | Fake/Bias, Real/Unbias      | N/A                      |
| Mistral-v0.3                | Fine-tuning                    | Decoder-only autoregressive sequence classification | Fake/Bias, Real/Unbias      | N/A                      |
| Mistral-large-instruct (IFT) | IFT                           | N/A                                          | Fake/Bias, Real/Unbias      | N/A                      |
| Gemma-2-9b-Instruct          | 0-shot, 5-shot, IFT            | Decoder-only, Causal LM                       | Fake/Bias, Real/Unbias      | N/A                      |
| Gemma-2-9b                  | Fine-tuning                    | Decoder-only, sequence classification         | Fake/Bias, Real/Unbias      | N/A                      |

## Benchmarking on Multimodality

### Small Language Models (SLMs)

| Model                              | Training Method      | Architecture (text-image)   |
|------------------------------------|----------------------|------------------|
| BERT + ResNet-34                   | Fine-tuning          | Encoder-Encoder   |
| SAFE (Text-CNN + Image2Sentence)   | Fine-tuning          | Encoder-Encoder   |
| SpotFake (XLNET + VGG-19)          | Fine-tuning          | Encoder-encoder   |
| MCAN (BERT + VGG-19/CNN)           | Fine-tuning          | Encoder-encoder   |
| FND-CLIP (BERT/ResNet + CLIP)      | Fine-tuning          | Encoder-encoder   |
| InstructBlipV                      | Fine-tuning          | Encoder-encoder   |
| DistilBERT + CLIP                  | Fine-tuning          | Encoder-encoder   |

### Large Language Models (LLMs)

| Model                                              | Training Method                          | Architecture (text-image)    |
|----------------------------------------------------|------------------------------------------|------------------|
| google/paligemma-3b-pt-224                         | Instruction fine-tuning                  | Decoder-encoder   |
| microsoft/Phi-3-vision-128k-instruct               | 0-shot, 5-shot, Instruction fine-tuning | Decoder-encoder   |
| Pixtral-12B-2409                                   | 0-shot, 5-shot, Instruction fine-tuning | Decoder-encoder   |
| LLaVA-1.6                                          | 0-shot, 5-shot, Instruction fine-tuning | Decoder-encoder   |
| Llama-3.2-11B-Vision-Instruct                      | 0-shot, 5-shot, Instruction fine-tuning | Decoder-encoder   |
| meta-llama/Llama-3.2-11B-Vision                    | Fine-tuning                              | Decoder-encoder   |
| meta-llama/Llama-Guard-3-11B-Vision                | Inference                                | Decoder-encoder   |