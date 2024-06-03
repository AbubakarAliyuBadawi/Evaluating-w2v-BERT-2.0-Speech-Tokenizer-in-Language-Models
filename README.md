# Evaluating w2v-BERT-2.0 Speech Tokenizer for Speech Models

## Overview

This project explores the integration of the W2v-BERT-2.0 tokenizer into existing speech-language models (SLMs) to enhance speech-to-speech translation capabilities. This approach aims to process speech directly, eliminating the need for converting speech to text first. Our focus is on leveraging the Conformer-based architecture of the W2v-BERT-2.0 tokenizer, extensively pre-trained on a multilingual corpus.

## Team

- Abubakar Aliyu Badawi
- Celil Yilmaz
- Tayyab Tahir
- Eshfaz Bhat

*Department of Seatech, University de Toulon, La Garde 83130, France*

## Project Aim

To assess the efficacy of the W2v-BERT-2.0 tokenizer integrated into an existing model, aimed at reproducing the open-source version of Google’s AudioLM. This integration is intended to improve the accuracy and efficiency of speech language processing directly, bypassing the intermediary step of text conversion.

## Key Objectives

1. Process audio data into TSV files containing training and testing datasets.
2. Explore the architecture and functionalities of the W2v-BERT-2.0 tokenizer.
3. Extract high-quality features from audio files using the tokenizer.
4. Train the enhanced model on extensive datasets, evaluating its performance against established benchmarks.

## Datasets

- **Train-clean-100**: 100 hours of high-quality, clean audio.
- **Libri-Light Large**: 60,000 hours of varied audio recordings, enhancing model robustness and adaptability.

## Methodology

- Pre-process audio data to prepare it for training.
- Identify the optimal layer within the W2v-BERT-2.0 model for audio feature extraction.
- Use K-means clustering for efficient categorization of speech patterns.
- Train and evaluate the model using scripts adapted for our specific setup, focusing on syntactic performance.

## Results

Our initial results indicate promising improvements in syntactic understanding and generalization capabilities of the speech model. Further training and optimization are recommended to fully realize the potential of the W2v-BERT-2.0 integration.

## Conclusion and Recommendations

While initial findings are encouraging, extended training periods and enhanced computational resources are suggested to further improve the model's performance.

## References

Detailed references are included for further reading and verification of the methodologies used.

---

For more information, please contact [abubakar-aliyu-badawi](mailto:abubakar-aliyu-badawi@etud.univ-tln.com).
