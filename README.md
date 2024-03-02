# SentinelLMs: Encrypted Input Adaptation and Fine-Tuning for Secure Inference

<p align="center">
    📃 <a href="https://arxiv.org/abs/2312.17342" target="_blank">[SentinelLMs@Conference]</a> • 📂 <a href="https://github.com/abhijitmishra/sentinellm-aaai2024/tree/main" target="_blank">[GitHub]</a>
</p>

## Introduction

SentinelLMs introduces a groundbreaking approach to address privacy and security concerns in deploying deep neural language models for AI applications. By enabling these models to perform inference on passkey-encrypted inputs, SentinelLMs ensures that user data remains private and secure, mitigating risks associated with data interception and storage.

This novel method involves a quick adaptation of pre-trained transformer-based language models, applying irreversible transformations to the tokenizer and token embeddings. This process allows the model to understand encrypted text without the possibility of reverse engineering. Following adaptation, the models are fine-tuned on encrypted versions of training datasets, ensuring performance parity with their unencrypted counterparts.

## Features

- **Privacy-Preserving**: Adapts language models to operate on encrypted inputs, ensuring user data privacy.
- **Secure Inference**: Prevents reverse engineering of text from model parameters and outputs.
- **Performance Parity**: Achieves comparable performance with original models on encrypted data.
- **Easy Integration**: Designed for easy integration with existing AI applications requiring language understanding.

## Getting Started

To begin using SentinelLMs for your secure inference needs, please follow the steps below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YourGitHubUsername/SentinelLMs.git
