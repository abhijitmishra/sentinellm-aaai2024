# SentinelLMs: Encrypted Input Adaptation and Fine-Tuning for Secure Inference 

<p align="center">
    📃 <a href="https://arxiv.org/abs/2312.17342" target="_blank">[SentinelLMs@AAAI2024]</a> • 📧 <a href="https://abhijitmishra.github.io/" 
        target="_blank">[Contact]</a>
</p>

## Introduction

We introduce **SentinelLM**, a groundbreaking approach to address privacy and security concerns in deploying deep neural language models for AI applications. By enabling these models to perform inference on **passkey-encrypted inputs**, we ensure that user data remains private and secure, mitigating risks associated with data interception and storage.

This novel method involves a quick adaptation of pre-trained transformer-based language models, applying irreversible transformations to the tokenizer and token embeddings. This process allows the model to understand encrypted text without the possibility of reverse engineering. Following adaptation, the models are fine-tuned on **encrypted versions of training datasets**, ensuring performance parity with their unencrypted counterparts.

You can easily replicate our paper with this repository.

## Features

- **Privacy-Preserving**: Adapts language models to operate on encrypted inputs, ensuring user data privacy.
- **Secure Inference**: Prevents reverse engineering of text from model parameters and outputs.
- **Performance Parity**: Achieves comparable performance with original models on encrypted data.
- **Easy Integration**: Designed for easy integration with existing AI applications requiring language understanding.

## Getting Started

To begin using SentinelLMs for your secure inference needs, please follow the steps below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/abhijitmishra/sentinellm-aaai2024.git
   ```
   
2. **Install Dependencies**
    
    Navigate to the project directory and install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. **Model Encryption**
    
    Use llm_converter.py to create an encrypted version of the language model:
    ```bash
    !python llm_converter.py --model your_model_name_here --output encrypted_model_name_here
    ```
    
4. **Fine-Tuning on NLP Benchmark**
    
    Fine-tune the encrypted language model on specific NLP tasks using the corresponding run_xx_blake2.py script. Here is an example for Named Entity Recognition (NER):
    ```bash
    !python run_ner_blake2.py \
        --model_name_or_path "Mingda/bert-base-uncased-blake2-glide3-key-llm123-shuffled" \
        --dataset_name conll2003 \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_eval \
        --encryption_key "llm123"
    ```
    For more detailed examples and information, please refer to the CameraReady branch in this repository under the Code section.

## Citation

If you find our work useful for your research, please consider citing it as follows:

```bibtex
@article{mishra2023sentinellms,
  title={SentinelLMs: Encrypted Input Adaptation and Fine-tuning of Language Models for Private and Secure Inference},
  author={Mishra, Abhijit and Li, Mingda and Deo, Soham},
  journal={arXiv preprint arXiv:2312.17342},
  year={2023}
}
