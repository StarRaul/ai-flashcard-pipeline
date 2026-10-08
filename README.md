# AI Flashcard Pipeline & MinLlama

A complete machine learning pipeline implemented in PyTorch. This repository covers the progression from foundational NLP and custom attention mechanics to LoRA fine-tuning, RAG, and an automated Anki flashcard generator.

## Project includes

### 1. NLP Basics (`NLPBasics`)
Text preprocessing pipelines, subword tokenization, and vocabulary mapping.

### 2. Language Modeling (`LanguageModeling`)
Next-token prediction, cross-entropy training, and autoregressive generation loops.

### 3. MinLlama (`MinLlama`)
A lightweight PyTorch implementation of the LLaMA architecture.
* Rotary Position Embeddings (RoPE)
* RMSNorm pre-normalization
* SwiGLU activations
* Causal self-attention

### 4. Fine-Tuning (`FineTuning`)
Parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) to train model weights efficiently.

### 5. RAG (`RAG`)
Text chunking, vector embeddings, and similarity search for context-grounded text generation.

### 6. Anki Card Generator (`GenAnkiCards`)
Document parser that takes technical text and outputs formatted `.apkg` study decks for spaced repetition.

