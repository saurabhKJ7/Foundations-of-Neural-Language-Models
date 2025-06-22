#!/usr/bin/env python3
"""
Quick test script to verify tokenization functionality
"""

def test_basic_tokenization():
    """Test basic tokenization without heavy model loading"""
    sentence = "The cat sat on the mat because it was tired."
    
    print("Testing basic tokenization...")
    print(f"Sentence: {sentence}")
    
    try:
        from transformers import AutoTokenizer
        
        # Test with a lightweight tokenizer
        print("\n1. Testing GPT-2 tokenizer (BPE):")
        gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokens = gpt2_tokenizer.tokenize(sentence)
        token_ids = gpt2_tokenizer.encode(sentence)
        print(f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids}")
        print(f"   Count: {len(tokens)}")
        
        print("\n2. Testing BERT tokenizer (WordPiece):")
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = bert_tokenizer.tokenize(sentence)
        token_ids = bert_tokenizer.encode(sentence)
        print(f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids}")
        print(f"   Count: {len(tokens)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_fill_mask():
    """Test fill-mask functionality"""
    print("\nTesting fill-mask prediction...")
    
    try:
        from transformers import pipeline
        
        masked_sentence = "The [MASK] sat on the mat."
        print(f"Masked sentence: {masked_sentence}")
        
        # Use a smaller model for testing
        fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
        predictions = fill_mask(masked_sentence)
        
        print("Predictions:")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"   {i}. {pred['token_str']} (score: {pred['score']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("TOKENIZATION TEST")
    print("=" * 50)
    
    success = True
    
    # Test tokenization
    if not test_basic_tokenization():
        success = False
    
    # Test fill-mask
    if not test_fill_mask():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 50)

if __name__ == "__main__":
    main()