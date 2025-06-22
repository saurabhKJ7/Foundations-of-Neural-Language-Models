# Q1: Tokenization & Fill-in-the-Blank Results

## Overview
This document presents the results from tokenizing the sentence **"The cat sat on the mat because it was tired."** using three different algorithms, followed by fill-in-the-blank predictions.

## 🔤 Tokenization Results

### BPE (GPT-2)
- **Tokens**: `['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']`
- **Token IDs**: `[464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032, 13]`
- **Total Count**: 11 tokens

### WordPiece (BERT)
- **Tokens**: `['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']`
- **Token IDs**: `[101, 1996, 4937, 2938, 2006, 1996, 13523, 2138, 2009, 2001, 5458, 1012, 102]`
- **Total Count**: 11 tokens

### SentencePiece (T5)
- **Tokens**: `['▁The', '▁cat', '▁', 's', 'at', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.']`
- **Token IDs**: `[37, 1712, 3, 7, 144, 30, 8, 6928, 250, 34, 47, 7718, 5, 1]`
- **Total Count**: 13 tokens

## 📋 Analysis: Why Tokenization Differs

The three algorithms produce different token splits due to their distinct approaches:

**BPE (Byte-Pair Encoding)**:
- Uses frequency-based merging of character pairs
- The 'Ġ' symbol represents spaces in GPT-2's implementation
- Maintains word-level splits for this simple sentence
- Efficient at handling subword patterns based on training data statistics

**WordPiece**:
- Employs greedy longest-match-first algorithm
- Produces clean word-level tokens without space markers
- Uses special tokens [CLS] and [SEP] (included in token IDs)
- Better at preserving complete words when they exist in vocabulary

**SentencePiece**:
- Uses unigram language model for probabilistic segmentation
- Treats spaces explicitly with '▁' symbol
- Splits "sat" into '▁', 's', 'at' showing subword decomposition
- Results in higher token count (13 vs 11) due to more aggressive splitting
- More flexible for languages without clear word boundaries

## 🧠 Fill-in-the-Blank Predictions

### Mask Example 1: "The [MASK] sat on the mat because it was tired."
**Top-3 Predictions**:
1. **'dog'** (confidence: 0.0998)
2. **'cat'** (confidence: 0.0728)
3. **'horse'** (confidence: 0.0608)

### Mask Example 2: "The cat sat on the [MASK] because it was tired."
**Top-3 Predictions**:
1. **'floor'** (confidence: 0.2675)
2. **'bed'** (confidence: 0.1229)
3. **'couch'** (confidence: 0.1124)

### Mask Example 3: "The cat sat on the mat because it was [MASK]."
**Top-3 Predictions**:
1. **'cold'** (confidence: 0.0683)
2. **'warm'** (confidence: 0.0395)
3. **'hungry'** (confidence: 0.0336)

## 💭 Plausibility Analysis

The BERT model's predictions demonstrate strong contextual understanding:

**First Mask (Subject)**: The model correctly identifies that the masked token should be an entity capable of sitting. Animals like 'dog', 'cat', and 'horse' are semantically appropriate. Interestingly, 'dog' scores higher than the original 'cat', suggesting the model finds dogs more commonly associated with this behavior pattern.

**Second Mask (Location)**: The predictions show excellent spatial reasoning. 'Floor' has the highest confidence (0.2675), which makes sense as a common surface for tired animals. 'Bed' and 'couch' are also plausible furniture items where tired beings rest.

**Third Mask (State)**: The model predicts physical/emotional states that would motivate sitting/resting behavior. While the original word was 'tired', the model suggests 'cold', 'warm', and 'hungry' - all states that could explain why an animal might sit and rest.

The bidirectional attention mechanism in BERT enables it to use both preceding and following context, resulting in semantically coherent predictions that maintain narrative consistency.

## Key Insights

1. **Tokenization Efficiency**: WordPiece and BPE achieved the same token count (11) while SentencePiece required more tokens (13) due to its character-level approach.

2. **Context Understanding**: BERT's predictions show sophisticated understanding of semantic roles, spatial relationships, and causal reasoning.

3. **Algorithm Trade-offs**: Each tokenization method optimizes for different criteria - frequency (BPE), greedy matching (WordPiece), and statistical likelihood (SentencePiece).

4. **Model Capabilities**: The fill-mask task demonstrates how pre-trained language models capture rich linguistic and world knowledge from their training data.