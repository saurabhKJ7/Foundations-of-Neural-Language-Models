import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, random_seed=42):
        """
        Initialize perceptron with random weights and bias
        """
        np.random.seed(random_seed)
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.accuracy_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function with clipping to prevent overflow"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias randomly"""
        self.weights = np.random.normal(0, 0.1, n_features)
        self.bias = np.random.normal(0, 0.1)
    
    def forward(self, X):
        """Forward pass: compute predictions"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_cost(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy"""
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def fit(self, X, y, epochs=500, verbose=True):
        """
        Train the perceptron using batch gradient descent
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        # Store initial random predictions for reflection
        initial_predictions = self.forward(X)
        initial_cost = self.compute_cost(y, initial_predictions)
        initial_accuracy = self.compute_accuracy(y, initial_predictions)
        
        if verbose:
            print(f"Initial random predictions:")
            print(f"Initial cost: {initial_cost:.4f}")
            print(f"Initial accuracy: {initial_accuracy:.4f}")
            print("\nStarting training...")
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute cost and accuracy
            cost = self.compute_cost(y, y_pred)
            accuracy = self.compute_accuracy(y, y_pred)
            
            # Store metrics
            self.cost_history.append(cost)
            self.accuracy_history.append(accuracy)
            
            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")
            
            # Early stopping if cost is low enough
            if cost < 0.05:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} - Cost threshold reached: {cost:.4f}")
                break
        
        # Final results
        final_predictions = self.forward(X)
        final_cost = self.compute_cost(y, final_predictions)
        final_accuracy = self.compute_accuracy(y, final_predictions)
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"Final cost: {final_cost:.4f}")
            print(f"Final accuracy: {final_accuracy:.4f}")
            print(f"Improvement in cost: {initial_cost - final_cost:.4f}")
            print(f"Improvement in accuracy: {final_accuracy - initial_accuracy:.4f}")
        
        return {
            'initial_cost': initial_cost,
            'initial_accuracy': initial_accuracy,
            'final_cost': final_cost,
            'final_accuracy': final_accuracy,
            'epochs_trained': len(self.cost_history)
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        return self.forward(X)

def load_and_preprocess_data(filepath):
    """Load and preprocess the fruit dataset"""
    df = pd.read_csv(filepath)
    
    # Features and labels
    X = df[['length_cm', 'weight_g', 'yellow_score']].values
    y = df['label'].values
    
    # Normalize features for better convergence
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X_normalized, y, df

def plot_training_metrics(cost_history, accuracy_history):
    """Plot training loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cost
    ax1.plot(cost_history, 'b-', linewidth=2)
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(accuracy_history, 'r-', linewidth=2)
    ax2.set_title('Training Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_learning_rate_impact():
    """Demonstrate the impact of different learning rates"""
    # Load data
    X, y, _ = load_and_preprocess_data('fruit.csv')
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    colors = ['blue', 'green', 'red', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        perceptron = Perceptron(learning_rate=lr, random_seed=42)
        results = perceptron.fit(X, y, epochs=500, verbose=False)
        
        plt.subplot(2, 2, 1)
        plt.plot(perceptron.cost_history, color=colors[i], label=f'LR={lr}', linewidth=2)
        plt.title('Loss vs Epochs for Different Learning Rates')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(perceptron.accuracy_history, color=colors[i], label=f'LR={lr}', linewidth=2)
        plt.title('Accuracy vs Epochs for Different Learning Rates')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the perceptron training and evaluation"""
    print("=== Perceptron From Scratch - Fruit Classification ===\n")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, df = load_and_preprocess_data('fruit.csv')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: length_cm, weight_g, yellow_score")
    print(f"Labels: {np.bincount(y)} (0=apple, 1=banana)")
    print(f"Dataset preview:")
    print(df.head())
    print()
    
    # Create and train perceptron
    print("Training perceptron...")
    perceptron = Perceptron(learning_rate=0.1, random_seed=42)
    results = perceptron.fit(X, y, epochs=500, verbose=True)
    
    # Plot training metrics
    print("\nPlotting training metrics...")
    plot_training_metrics(perceptron.cost_history, perceptron.accuracy_history)
    
    # Test on training data (for demonstration)
    print("\nFinal predictions on training data:")
    predictions = perceptron.predict(X)
    probabilities = perceptron.predict_proba(X)
    
    for i in range(len(y)):
        fruit_type = "banana" if y[i] == 1 else "apple"
        pred_type = "banana" if predictions[i] == 1 else "apple"
        correct = "✓" if predictions[i] == y[i] else "✗"
        print(f"Sample {i+1}: True={fruit_type}, Pred={pred_type} ({probabilities[i]:.3f}) {correct}")
    
    # Demonstrate learning rate impact
    print("\nDemonstrating learning rate impact...")
    demonstrate_learning_rate_impact()
    
    print("\nTraining completed! Check the generated plots for visualization.")

if __name__ == "__main__":
    main()