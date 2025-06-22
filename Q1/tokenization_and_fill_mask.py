#!/usr/bin/env python3
"""
Q1: Tokenization & Fill-in-the-Blank Implementation
This script demonstrates different tokenization algorithms and mask prediction using transformers.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BertTokenizer, GPT2Tokenizer
)
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
import os
import tempfile

def setup_bpe_tokenizer():
    """Setup BPE tokenizer"""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer

def setup_wordpiece_tokenizer():
    """Setup WordPiece tokenizer"""
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer

def setup_sentencepiece_tokenizer():
    """Setup SentencePiece (Unigram) tokenizer"""
    # Create a temporary file for training data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # Add some sample text for training (in practice, you'd use a large corpus)
        sample_text = """
        The cat sat on the mat because it was tired.
        The dog ran in the park.
        A bird flew over the house.
        The sun shines brightly today.
        I love reading books in the library.
        """ * 100  # Repeat to have enough data
        f.write(sample_text)
        temp_file = f.name
    
    # Train SentencePiece model
    model_prefix = tempfile.mktemp()
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=1000,
        model_type='unigram',
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )
    
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    # Cleanup
    os.unlink(temp_file)
    os.unlink(f'{model_prefix}.model')
    os.unlink(f'{model_prefix}.vocab')
    
    return sp

def tokenize_with_pretrained_models(sentence):
    """Tokenize using pre-trained models for comparison"""
    results = {}
    
    # BPE-style tokenization using GPT-2
    try:
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = gpt2_tokenizer.tokenize(sentence)
        token_ids = gpt2_tokenizer.encode(sentence)
        results['BPE (GPT-2)'] = {
            'tokens': tokens,
            'token_ids': token_ids,
            'count': len(tokens)
        }
    except Exception as e:
        print(f"Error with GPT-2 tokenizer: {e}")
    
    # WordPiece tokenization using BERT
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = bert_tokenizer.tokenize(sentence)
        token_ids = bert_tokenizer.encode(sentence)
        results['WordPiece (BERT)'] = {
            'tokens': tokens,
            'token_ids': token_ids,
            'count': len(tokens)
        }
    except Exception as e:
        print(f"Error with BERT tokenizer: {e}")
    
    # SentencePiece tokenization using T5
    try:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        tokens = t5_tokenizer.tokenize(sentence)
        token_ids = t5_tokenizer.encode(sentence)
        results['SentencePiece (T5)'] = {
            'tokens': tokens,
            'token_ids': token_ids,
            'count': len(tokens)
        }
    except Exception as e:
        print(f"Error with T5 tokenizer: {e}")
    
    return results

def create_masked_sentence(sentence, tokenizer, mask_positions=[2, 8]):
    """Create a masked version of the sentence"""
    tokens = tokenizer.tokenize(sentence)
    
    # Ensure mask positions are valid
    valid_positions = [pos for pos in mask_positions if pos < len(tokens)]
    
    # Create masked tokens
    masked_tokens = tokens.copy()
    for pos in valid_positions:
        masked_tokens[pos] = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else '<mask>'
    
    return tokenizer.convert_tokens_to_string(masked_tokens), valid_positions

def fill_mask_prediction(masked_sentence, model_name='bert-base-uncased'):
    """Perform fill-mask prediction"""
    try:
        # Use fill-mask pipeline
        fill_mask = pipeline('fill-mask', model=model_name, tokenizer=model_name)
        predictions = fill_mask(masked_sentence)
        return predictions
    except Exception as e:
        print(f"Error with fill-mask prediction: {e}")
        return None

def analyze_tokenization_differences():
    """Provide analysis of why tokenization algorithms differ"""
    analysis = """
    Tokenization Algorithm Differences:
    
    1. BPE (Byte-Pair Encoding): Merges the most frequent pairs of bytes/characters iteratively.
       - Tends to create subword units based on frequency
       - Good at handling out-of-vocabulary words
       - May split common words into subwords if they contain frequent character pairs
    
    2. WordPiece: Similar to BPE but uses a greedy longest-match-first algorithm.
       - Prioritizes longer subwords when possible
       - Uses ## prefix for continuation tokens
       - Better at preserving word boundaries than BPE
    
    3. SentencePiece (Unigram): Uses a probabilistic approach with unigram language model.
       - Treats whitespace as a regular character
       - More flexible segmentation based on statistical likelihood
       - Can handle languages without clear word boundaries better
    
    These differences lead to varying token splits because each algorithm optimizes for different criteria:
    frequency (BPE), greedy matching (WordPiece), and statistical likelihood (Unigram).
    """
    return analysis

def main():
    """Main function to run the tokenization and fill-mask tasks"""
    sentence = "The cat sat on the mat because it was tired."
    
    print("=" * 80)
    print("Q1: TOKENIZATION & FILL-IN-THE-BLANK")
    print("=" * 80)
    print(f"Original sentence: {sentence}")
    print()
    
    # Task 1: Tokenization
    print("🔤 TOKENIZATION RESULTS")
    print("-" * 40)
    
    tokenization_results = tokenize_with_pretrained_models(sentence)
    
    for method, result in tokenization_results.items():
        print(f"\n{method}:")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Token IDs: {result['token_ids'][:10]}{'...' if len(result['token_ids']) > 10 else ''}")
        print(f"  Total tokens: {result['count']}")
    
    print("\n📋 ANALYSIS:")
    print(analyze_tokenization_differences())
    
    # Task 2: Fill-in-the-Blank
    print("\n🧠 FILL-IN-THE-BLANK PREDICTION")
    print("-" * 40)
    
    # Use BERT for masking (it has mask tokens)
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create masked sentence - mask "cat" and "tired"
        masked_sentence = sentence.replace("cat", "[MASK]").replace("tired", "[MASK]")
        print(f"Masked sentence: {masked_sentence}")
        
        # Get predictions
        predictions = fill_mask_prediction(masked_sentence, 'bert-base-uncased')
        
        if predictions:
            print("\nTop-3 predictions per mask:")
            
            # Group predictions by mask position
            mask_predictions = {}
            for pred in predictions:
                sequence = pred['sequence']
                if sequence not in mask_predictions:
                    mask_predictions[sequence] = []
                mask_predictions[sequence].append(pred)
            
            for i, (sequence, preds) in enumerate(mask_predictions.items()):
                print(f"\nMask {i+1}: {sequence}")
                for j, pred in enumerate(preds[:3]):
                    print(f"  {j+1}. {pred['token_str']} (confidence: {pred['score']:.4f})")
        
        print("\n💭 PLAUSIBILITY COMMENT:")
        print("""
        The predictions show semantic understanding:
        - For the first mask (originally 'cat'), predictions likely include animals or entities that can sit
        - For the second mask (originally 'tired'), predictions should include states or emotions
        - Higher confidence scores indicate better contextual fit based on the model's training
        - BERT's bidirectional attention allows it to use both left and right context for predictions
        """)
        
    except Exception as e:
        print(f"Error in fill-mask task: {e}")
        print("Note: This might require installing additional dependencies or model downloads")
    
    print("\n" + "=" * 80)
    print("TASK COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()