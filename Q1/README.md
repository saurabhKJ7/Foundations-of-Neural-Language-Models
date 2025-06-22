# Q1: Tokenization & Fill-in-the-Blank

This directory contains the implementation for Q1, demonstrating different tokenization algorithms and fill-in-the-blank prediction using transformer models.

## Overview

The implementation covers:
1. **Tokenization** using BPE, WordPiece, and SentencePiece algorithms
2. **Fill-in-the-Blank** prediction using masked language models
3. **Analysis** of tokenization differences and prediction plausibility

## Files

- `simple_tokenization_demo.py` - Main demonstration script (recommended)
- `tokenization_and_fill_mask.py` - Comprehensive implementation
- `setup.py` - Installation helper script
- `requirements.txt` - Package dependencies
- `README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
# Option 1: Use the setup script
python setup.py

# Option 2: Install manually
pip install torch transformers tokenizers sentencepiece numpy accelerate
```

### 2. Run the Demo

```bash
python simple_tokenization_demo.py
```

## What It Does

### Tokenization Comparison
The script tokenizes the sentence: **"The cat sat on the mat because it was tired."**

Using three different algorithms:
- **BPE (GPT-2)**: Byte-Pair Encoding with frequency-based merging
- **WordPiece (BERT)**: Greedy longest-match with ## continuation tokens  
- **SentencePiece (T5)**: Unigram model with probabilistic segmentation

### Fill-in-the-Blank Prediction
Creates masked versions of the sentence and uses BERT to predict:
- `The [MASK] sat on the mat because it was tired.`
- `The cat sat on the [MASK] because it was tired.`
- `The cat sat on the mat because it was [MASK].`

## Expected Output

### Tokenization Results
Each algorithm produces different token splits:
- **Token lists** showing how the sentence is segmented
- **Token IDs** representing numerical encoding
- **Token counts** showing efficiency differences

### Analysis
Explanation of why algorithms differ:
- Training objectives (frequency vs. likelihood vs. greedy matching)
- Whitespace handling approaches
- Vocabulary construction methods

### Predictions
Top-3 predictions for each masked position with:
- Confidence scores
- Semantic plausibility analysis
- Commentary on model understanding

## Key Insights

1. **BPE** tends to create subword units based on character pair frequency
2. **WordPiece** preserves word boundaries better with greedy matching
3. **SentencePiece** handles whitespace explicitly and works well across languages
4. **BERT's bidirectional** attention enables contextually appropriate predictions

## Troubleshooting

### Common Issues

**Model Download Errors**: 
- Ensure internet connection for first-time model downloads
- Models are cached locally after first download

**Memory Issues**:
- Large models require significant RAM
- Consider using smaller model variants if needed

**Import Errors**:
- Run `python setup.py` to install all dependencies
- Verify Python version compatibility (3.8+)

### Alternative Models

If default models don't work, you can modify the script to use:
- `distilbert-base-uncased` (smaller BERT variant)
- `gpt2-medium` (different BPE implementation)
- `t5-base` (larger T5 variant)

## Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Tokenizers**: Fast tokenization library
- **SentencePiece**: Google's text processing library

### Model Requirements
- Internet connection for initial downloads
- ~500MB-2GB storage for cached models
- 4GB+ RAM recommended for larger models

## Learning Objectives

This implementation demonstrates:
- Differences between major tokenization algorithms
- How subword tokenization handles vocabulary limitations
- Masked language model capabilities
- Practical use of transformer libraries

## Next Steps

After running this demo, consider:
- Experimenting with different sentences
- Trying other tokenization libraries
- Exploring multilingual tokenization
- Implementing custom tokenization schemes