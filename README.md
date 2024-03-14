# Speech Language Models DDMP

## Overview
This project is part of the "Data-driven Machine Perception" (DDMP) course and focuses on building spoken dialogue systems directly from raw audio recordings. Our approach, inspired by the Zero Resource Speech Challenge, aims to support low-resource languages and explore expressive communication methods without relying on text.

## Motivations
- **Inclusivity**: Targeting low-resource languages to make AI dialogue systems more accessible.
- **Expressiveness**: Leveraging the richness of audio for more nuanced interactions.
- **Language Acquisition Insights**: Offering insights into human language understanding.

## Research and Implementation
We're building upon Google’s open-source AudioLM, focusing on:
- Evaluating speech tokenizers.
- Improving acoustic quality measurements.
- Extending to music generation.

# Speech-to-Speech Model Enhancement Project

## Overview
This project is part of the Data-driven Machine Perception course, focusing on enhancing speech language models. Our primary objective is to improve upon an open-source version of Google's AudioLM, a top speech-to-speech (S2S) model, by integrating a more advanced tokenizer. The original model architecture includes a speech tokenizer, a speech language model with a GPT-like structure, and a token-to-speech model to convert tokens back into speech. We aim to replace the tokenizer component with the recently released W2v-Bert-2.0 from Hugging Face, comparing its performance against the original setup.

## Project Goals
- Integrate W2v-Bert-2.0 as the new tokenizer in the existing S2S model architecture.
- Determine the optimal layer of W2v-Bert-2.0 for feature extraction through empirical analysis.
- Compare the performance of the enhanced model with the original version to evaluate improvements.

## Progress Summary

### 1. Data Pre-processing
- Split 100 hours of audio files into training and testing datasets.
- Organized datasets into a TSV (Tab Separated Values) file for easier access and management.

### 2. Feature Extraction with W2v-Bert-2.0
- Analyzed the W2v-Bert-2.0 model, which consists of 24 layers and operates at a sampling rate of 50Hz.
- To identify the most effective layer for feature extraction, we trained a linear classifier on audio files and their corresponding labels (phones).
- Results indicated that layer 16 outperforms the others, achieving an accuracy of 86% in feature representation.

### 3. Implementation of Layer 16 for Feature Extraction
**Successfully integrated layer 16 of W2v-Bert-2.0** 
- within the speech-to-speech model's framework for feature extraction.
- This enhancement is pivotal to the tokenizer replacement process, setting the stage for comprehensive performance evaluations against the model's original configuration.

## Next Steps
- Implement layer 16 of W2v-Bert-2.0 for feature extraction within the speech-to-speech model framework.
- Integrate the extracted features into the speech language model to complete the tokenizer replacement.
- Conduct extensive testing to compare the performance of the enhanced model against the original version.
- Analyze results and document findings, focusing on any improvements in accuracy or efficiency brought by the new tokenizer.

## Resources
- [W2v-Bert-2.0 on Hugging Face](https://huggingface.co/facebook/w2v-bert-2.0)


