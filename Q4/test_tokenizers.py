"""
Test script to verify tokenization implementations work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tokenization_demo import SimpleTokenizer, SimpleBPETokenizer


def test_simple_tokenizer():
    """Test the character-level tokenizer"""
    print("Testing SimpleTokenizer...")
    
    tokenizer = SimpleTokenizer()
    texts = ["hello world", "test text"]
    tokenizer.build_vocab(texts)
    
    # Test encoding/decoding
    test_text = "hello"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == test_text, f"Expected '{test_text}', got '{decoded}'"
    assert len(tokenizer.vocab) > 0, "Vocabulary should not be empty"
    
    print("✓ SimpleTokenizer tests passed")


def test_bpe_tokenizer():
    """Test the BPE tokenizer"""
    print("Testing SimpleBPETokenizer...")
    
    tokenizer = SimpleBPETokenizer(vocab_size=50)
    texts = ["hello world", "world peace", "hello peace"]
    tokenizer.train(texts)
    
    # Test tokenization
    test_text = "hello world"
    tokens = tokenizer.tokenize(test_text)
    
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) > 0, "Should produce at least one token"
    assert len(tokenizer.merges) > 0, "Should learn some merges"
    
    print("✓ SimpleBPETokenizer tests passed")


def test_tokenization_comparison():
    """Test that different tokenization methods produce different results"""
    print("Testing tokenization comparison...")
    
    text = "hello world"
    
    # Character level
    char_tokens = list(text)
    
    # Word level
    word_tokens = text.split()
    
    # Different lengths expected
    assert len(char_tokens) > len(word_tokens), "Character tokens should be more numerous"
    
    print("✓ Tokenization comparison tests passed")


def run_all_tests():
    """Run all tests"""
    print("Running tokenization tests...\n")
    
    try:
        test_simple_tokenizer()
        test_bpe_tokenizer()
        test_tokenization_comparison()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)