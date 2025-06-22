"""
Tokenization Demonstration Script
Practical examples and visualizations for the blog post on tokenization in LLMs
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """Basic character-level tokenizer for demonstration"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Add characters
        for i, char in enumerate(sorted(chars)):
            self.vocab[char] = i + 4
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        return [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return ''.join([self.reverse_vocab.get(id_, '<UNK>') for id_ in token_ids])


class SimpleBPETokenizer:
    """Simplified Byte Pair Encoding tokenizer"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        
    def get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts"""
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freqs[word] += 1
        return dict(word_freqs)
    
    def get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """Get statistics of symbol pairs"""
        pairs = defaultdict(int)
        for word, word_split in splits.items():
            for i in range(len(word_split) - 1):
                pairs[(word_split[i], word_split[i + 1])] += 1
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair"""
        new_splits = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in splits:
            new_word = ' '.join(splits[word])
            new_word = new_word.replace(bigram, replacement)
            new_splits[word] = new_word.split()
        return new_splits
    
    def train(self, texts: List[str]):
        """Train BPE tokenizer"""
        # Get word frequencies
        word_freqs = self.get_word_freqs(texts)
        
        # Initialize splits (each character is a token)
        splits = {}
        for word in word_freqs:
            splits[word] = list(word)
        
        # Build vocabulary
        vocab = set()
        for word in splits:
            vocab.update(splits[word])
        
        # Perform merges
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(splits)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            splits = self.merge_vocab(best_pair, splits)
            self.merges.append(best_pair)
            
        # Build final vocabulary
        self.vocab = set()
        for word in splits:
            self.vocab.update(splits[word])
        self.vocab = {token: i for i, token in enumerate(sorted(self.vocab))}
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned merges"""
        words = text.lower().split()
        splits = {}
        
        for word in words:
            splits[word] = list(word)
            
        for pair in self.merges:
            splits = self.merge_vocab(pair, splits)
            
        tokens = []
        for word in words:
            if word in splits:
                tokens.extend(splits[word])
            else:
                tokens.extend(list(word))
                
        return tokens


def demonstrate_tokenization_differences():
    """Show how different tokenization affects the same text"""
    text = "The unhappiness of artificial intelligence researchers"
    
    # Character-level
    char_tokens = list(text)
    
    # Word-level
    word_tokens = text.split()
    
    # Subword-level (simulated)
    subword_tokens = ["The", " un", "happiness", " of", " artificial", 
                     " intelligence", " research", "ers"]
    
    print("=== Tokenization Comparison ===")
    print(f"Original text: '{text}'")
    print(f"Character-level ({len(char_tokens)} tokens): {char_tokens}")
    print(f"Word-level ({len(word_tokens)} tokens): {word_tokens}")
    print(f"Subword-level ({len(subword_tokens)} tokens): {subword_tokens}")
    print()


def visualize_tokenization_efficiency():
    """Visualize token efficiency across different methods"""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence and machine learning",
        "Natural language processing with transformers",
        "Deep neural networks for computer vision",
        "Reinforcement learning algorithms"
    ]
    
    # Count tokens for each method
    char_counts = []
    word_counts = []
    
    for text in texts:
        char_counts.append(len(text))
        word_counts.append(len(text.split()))
    
    # Simulated subword counts (typically between char and word level)
    subword_counts = [int(c * 0.3 + w * 0.7) for c, w in zip(char_counts, word_counts)]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Token count comparison
    x = np.arange(len(texts))
    width = 0.25
    
    ax1.bar(x - width, char_counts, width, label='Character-level', alpha=0.8)
    ax1.bar(x, word_counts, width, label='Word-level', alpha=0.8)
    ax1.bar(x + width, subword_counts, width, label='Subword-level', alpha=0.8)
    
    ax1.set_xlabel('Text Samples')
    ax1.set_ylabel('Number of Tokens')
    ax1.set_title('Token Count Comparison Across Methods')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Text {i+1}' for i in range(len(texts))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency visualization
    avg_char = np.mean(char_counts)
    avg_word = np.mean(word_counts)
    avg_subword = np.mean(subword_counts)
    
    methods = ['Character', 'Subword', 'Word']
    efficiencies = [avg_char, avg_subword, avg_word]
    colors = ['red', 'green', 'blue']
    
    ax2.bar(methods, efficiencies, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Token Count')
    ax2.set_title('Average Tokenization Efficiency')
    ax2.grid(True, alpha=0.3)
    
    # Add efficiency annotations
    for i, (method, eff) in enumerate(zip(methods, efficiencies)):
        ax2.annotate(f'{eff:.1f}', (i, eff + 1), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tokenization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_bpe_training():
    """Show BPE training process step by step"""
    texts = [
        "the quick brown fox",
        "the lazy dog runs",
        "quick brown animals",
        "lazy fox runs fast"
    ]
    
    print("=== BPE Training Demonstration ===")
    
    # Initialize BPE tokenizer
    bpe = SimpleBPETokenizer(vocab_size=50)
    bpe.train(texts)
    
    print(f"Learned {len(bpe.merges)} merges:")
    for i, merge in enumerate(bpe.merges[:10]):  # Show first 10 merges
        print(f"  {i+1}: {merge[0]} + {merge[1]} → {''.join(merge)}")
    
    print(f"\nFinal vocabulary size: {len(bpe.vocab)}")
    
    # Test tokenization
    test_text = "the quick lazy fox"
    tokens = bpe.tokenize(test_text)
    print(f"\nTokenizing: '{test_text}'")
    print(f"Result: {tokens}")
    print()


def show_tokenization_failures():
    """Demonstrate common tokenization failure cases"""
    print("=== Common Tokenization Challenges ===")
    
    # Character counting issue
    text = "strawberry"
    simulated_tokens = ["straw", "berry"]
    print(f"Text: '{text}'")
    print(f"Tokenized as: {simulated_tokens}")
    print(f"Issue: Model sees 2 tokens, not {len(text)} characters")
    print("Impact: Difficulty with character-level tasks like reversal")
    print()
    
    # Number tokenization inconsistency
    numbers = ["1234", "5678", "9999", "1000"]
    # Simulate inconsistent number tokenization
    tokenized_numbers = [["12", "34"], ["567", "8"], ["99", "99"], ["1000"]]
    
    print("Number tokenization inconsistencies:")
    for num, tokens in zip(numbers, tokenized_numbers):
        print(f"  '{num}' → {tokens}")
    print("Impact: Inconsistent arithmetic reasoning")
    print()
    
    # Multilingual challenges
    print("Multilingual tokenization:")
    multilingual_examples = {
        "English": "hello world",
        "French": "bonjour monde", 
        "German": "hallo welt",
        "Japanese": "こんにちは世界"
    }
    
    for lang, text in multilingual_examples.items():
        # Simulate different tokenization patterns
        if lang == "Japanese":
            tokens = list(text)  # Character-level for non-Latin scripts
        else:
            tokens = text.split()  # Word-level for Latin scripts
        print(f"  {lang}: '{text}' → {tokens}")
    print()


def visualize_vocabulary_growth():
    """Show how vocabulary grows with different tokenization strategies"""
    # Simulate vocabulary growth
    corpus_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    
    # Character-level: grows slowly, plateaus early
    char_vocab = [min(100, 20 + np.log(size) * 5) for size in corpus_sizes]
    
    # Word-level: grows rapidly, no plateau
    word_vocab = [size * 0.1 for size in corpus_sizes]
    
    # Subword-level: controlled growth
    subword_vocab = [min(50000, size * 0.05) for size in corpus_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(corpus_sizes, char_vocab, 'r-o', label='Character-level', linewidth=2)
    plt.plot(corpus_sizes, word_vocab, 'b-s', label='Word-level', linewidth=2)
    plt.plot(corpus_sizes, subword_vocab, 'g-^', label='Subword-level', linewidth=2)
    
    plt.xlabel('Corpus Size (tokens)')
    plt.ylabel('Vocabulary Size')
    plt.title('Vocabulary Growth with Corpus Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('vocabulary_growth.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run all demonstrations"""
    print("🔤 Tokenization: The Hidden Engine of LLMs - Demo Script\n")
    print("=" * 60)
    
    # Basic tokenization comparison
    demonstrate_tokenization_differences()
    
    # BPE training demonstration
    demonstrate_bpe_training()
    
    # Show common failure cases
    show_tokenization_failures()
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_tokenization_efficiency()
    visualize_vocabulary_growth()
    
    print("Demo complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()