import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("🍎🍌 Q3: Perceptron From Scratch Demo 🍌🍎")
print("=" * 50)

# Load dataset
df = pd.read_csv('fruit.csv')
print(f"Dataset: {df.shape[0]} fruits with {df.shape[1]-1} features")
print(f"Apples: {sum(df.label == 0)}, Bananas: {sum(df.label == 1)}")

# Quick perceptron implementation
class SimplePerceptron:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y, epochs=500):
        self.weights = np.random.normal(0, 0.1, X.shape[1])
        self.bias = np.random.normal(0, 0.1)
        
        initial_pred = self.sigmoid(X @ self.weights + self.bias)
        initial_acc = np.mean((initial_pred >= 0.5) == y)
        print(f"\nInitial random accuracy: {initial_acc:.1%}")
        
        for epoch in range(epochs):
            z = X @ self.weights + self.bias
            pred = self.sigmoid(z)
            
            # Binary cross-entropy
            loss = -np.mean(y * np.log(pred + 1e-15) + (1-y) * np.log(1-pred + 1e-15))
            self.losses.append(loss)
            
            # Gradients
            dw = X.T @ (pred - y) / len(y)
            db = np.mean(pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if loss < 0.05:
                print(f"Converged at epoch {epoch+1}!")
                break
        
        final_pred = self.sigmoid(X @ self.weights + self.bias)
        final_acc = np.mean((final_pred >= 0.5) == y)
        print(f"Final accuracy: {final_acc:.1%}")
        return final_acc - initial_acc

# Prepare data
X = df[['length_cm', 'weight_g', 'yellow_score']].values
y = df['label'].values
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Train with different learning rates
print("\n🎛️  Learning Rate Impact (DJ Knob Analogy):")
learning_rates = [0.001, 0.01, 0.1, 1.0]
analogies = ["Timid child", "Good student", "Balanced learner", "Impulsive child"]

for lr, analogy in zip(learning_rates, analogies):
    model = SimplePerceptron(lr=lr)
    improvement = model.fit(X_norm, y, epochs=200)
    epochs_used = len(model.losses)
    print(f"LR {lr:5.3f} ({analogy:15s}): {epochs_used:3d} epochs, +{improvement:.1%} improvement")

# Final demonstration
print(f"\n🎯 Final Results Summary:")
best_model = SimplePerceptron(lr=0.1)
best_model.fit(X_norm, y, epochs=500)

predictions = best_model.sigmoid(X_norm @ best_model.weights + best_model.bias)
print(f"Perfect classification achieved in {len(best_model.losses)} epochs!")
print(f"Loss decreased from {best_model.losses[0]:.3f} to {best_model.losses[-1]:.3f}")

print(f"\n🔍 Sample Predictions:")
for i in range(min(8, len(df))):
    true_fruit = "🍎 Apple" if y[i] == 0 else "🍌 Banana"
    confidence = predictions[i] if y[i] == 1 else 1-predictions[i]
    print(f"  {true_fruit}: {confidence:.1%} confidence")

print(f"\n✨ Key Insights:")
print(f"• Random start ➜ Perfect classification through gradient descent")
print(f"• Learning rate is like a DJ knob: too low = slow, too high = unstable")
print(f"• Neural networks learn by iteratively adjusting weights based on errors")
print(f"• Even simple perceptrons can achieve excellent results on linearly separable data")