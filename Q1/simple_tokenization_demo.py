#!/usr/bin/env python3
"""
Q1: Tokenization & Fill-in-the-Blank - Simplified Demo
"""

import torch
from transformers import (
    AutoTokenizer, 
    BertTokenizer, 
    GPT2Tokenizer,
    T5Tokenizer,
    pipeline
)

def demonstrate_tokenization():
    """Demonstrate different tokenization algorithms"""
    sentence = "The cat sat on the mat because it was tired."
    
    print("=" * 80)
    print("🔤 TOKENIZATION COMPARISON")
    print("=" * 80)
    print(f"Original sentence: {sentence}")
    print()
    
    tokenizers_info = [
        ("BPE (GPT-2)", "gpt2", GPT2Tokenizer),
        ("WordPiece (BERT)", "bert-base-uncased", BertTokenizer), 
        ("SentencePiece (T5)", "t5-small", T5Tokenizer)
    ]
    
    results = {}
    
    for name, model_name, tokenizer_class in tokenizers_info:
        try:
            print(f"Loading {name}...")
            tokenizer = tokenizer_class.from_pretrained(model_name)
            
            # Tokenize
            tokens = tokenizer.tokenize(sentence)
            token_ids = tokenizer.encode(sentence, add_special_tokens=True)
            
            results[name] = {
                'tokens': tokens,
                'token_ids': token_ids,
                'count': len(tokens)
            }
            
            print(f"✅ {name}:")
            print(f"   Tokens: {tokens}")
            print(f"   Token IDs: {token_ids}")
            print(f"   Count: {len(tokens)} tokens")
            print()
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            print()
    
    return results

def analyze_differences():
    """Analyze why tokenization differs across algorithms"""
    print("📋 TOKENIZATION ANALYSIS")
    print("-" * 40)
    
    analysis = """
Why tokenization algorithms produce different results:

1. BPE (Byte-Pair Encoding):
   - Merges most frequent character/byte pairs iteratively
   - Creates subword vocabulary based on frequency statistics
   - May split common words if they contain frequent subword patterns
   - Example: "tired" might become "ti" + "red" if those pairs are frequent

2. WordPiece:
   - Uses greedy longest-match-first algorithm
   - Prefixes continuation pieces with "##"
   - Tries to preserve whole words when possible
   - Better at maintaining word boundaries than BPE
   - Example: "tired" stays as one token or becomes "tired" + "##d"

3. SentencePiece (Used in T5):
   - Treats text as sequence of Unicode characters including spaces
   - Uses Unigram language model for probabilistic segmentation
   - More flexible with languages that don't use spaces
   - Example: May tokenize "▁The" (space + The) as single unit

Key differences result from:
- Training objectives (frequency vs. likelihood vs. greedy matching)
- Handling of whitespace (explicit vs. implicit)
- Vocabulary construction methods
- Language-specific optimizations
"""
    
    print(analysis)

def demonstrate_fill_mask():
    """Demonstrate fill-in-the-blank prediction"""
    print("\n🧠 FILL-IN-THE-BLANK DEMONSTRATION")
    print("-" * 40)
    
    original = "The cat sat on the mat because it was tired."
    
    # Create different masked versions
    masked_sentences = [
        "The [MASK] sat on the mat because it was tired.",
        "The cat sat on the [MASK] because it was tired.", 
        "The cat sat on the mat because it was [MASK]."
    ]
    
    try:
        print("Loading BERT for fill-mask task...")
        fill_mask = pipeline('fill-mask', model='bert-base-uncased')
        
        for i, masked_sentence in enumerate(masked_sentences, 1):
            print(f"\nMask Example {i}: {masked_sentence}")
            
            predictions = fill_mask(masked_sentence)
            
            print("Top-3 predictions:")
            for j, pred in enumerate(predictions[:3], 1):
                token = pred['token_str']
                confidence = pred['score']
                print(f"  {j}. '{token}' (confidence: {confidence:.4f})")
        
        print("\n💭 PLAUSIBILITY ANALYSIS:")
        plausibility_comment = """
The predictions demonstrate the model's contextual understanding:

1. First mask (animal position): Likely predicts animals or entities that can "sit"
   - High confidence for semantically appropriate animals
   - Lower confidence for abstract concepts

2. Second mask (location): Should predict surfaces or locations suitable for sitting
   - Common furniture items (chair, floor, couch)
   - Outdoor locations (ground, grass)

3. Third mask (state/emotion): Predicts states that explain resting behavior
   - Emotional states (tired, sleepy, comfortable)
   - Physical conditions (sick, hurt, relaxed)

The bidirectional nature of BERT allows it to use both left and right context,
leading to semantically coherent predictions that fit the sentence structure.
"""
        print(plausibility_comment)
        
    except Exception as e:
        print(f"❌ Error in fill-mask demonstration: {e}")
        print("This might require internet connection for model download.")

def main():
    """Main execution function"""
    print("Q1: TOKENIZATION & FILL-IN-THE-BLANK")
    print("This demo requires internet connection for downloading models.")
    print()
    
    # Demonstrate tokenization
    tokenization_results = demonstrate_tokenization()
    
    # Analyze differences
    analyze_differences()
    
    # Demonstrate fill-mask
    demonstrate_fill_mask()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()