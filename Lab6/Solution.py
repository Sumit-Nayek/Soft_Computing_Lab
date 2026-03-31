import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Setup Data: 2 Input Variables, Binary Classification
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
y = y.reshape(-1, 1)
X = (X - X.mean(axis=0)) / X.std(axis=0) # Normalization

# 2. Optimizer Definitions
class Optimizers:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.v_w, self.v_b = 0, 0        # For Momentum/Adam
        self.s_w, self.s_b = 0, 0        # For AdaGrad/RMSprop/Adam
        self.t = 0                       # For Adam bias correction

    def gd(self, w, b, dw, db):
        return w - self.lr * dw, b - self.lr * db

    def momentum(self, w, b, dw, db, beta=0.9):
        self.v_w = beta * self.v_w + (1 - beta) * dw
        self.v_b = beta * self.v_b + (1 - beta) * db
        return w - self.lr * self.v_w, b - self.lr * self.v_b

    def adagrad(self, w, b, dw, db, eps=1e-8):
        self.s_w += dw**2
        self.s_b += db**2
        return w - (self.lr / np.sqrt(self.s_w + eps)) * dw, b - (self.lr / np.sqrt(self.s_b + eps)) * db

    def rmsprop(self, w, b, dw, db, beta=0.9, eps=1e-8):
        self.s_w = beta * self.s_w + (1 - beta) * (dw**2)
        self.s_b = beta * self.s_b + (1 - beta) * (db**2)
        return w - (self.lr / np.sqrt(self.s_w + eps)) * dw, b - (self.lr / np.sqrt(self.s_b + eps)) * db

    def adam(self, w, b, dw, db, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.v_w = b1 * self.v_w + (1 - b1) * dw
        self.v_b = b1 * self.v_b + (1 - b1) * db
        self.s_w = b2 * self.s_w + (1 - b2) * (dw**2)
        self.s_b = b2 * self.s_b + (1 - b2) * (db**2)
        m_w_hat = self.v_w / (1 - b1**self.t)
        m_b_hat = self.v_b / (1 - b1**self.t)
        v_w_hat = self.s_w / (1 - b2**self.t)
        v_b_hat = self.s_b / (1 - b2**self.t)
        return w - (self.lr / (np.sqrt(v_w_hat) + eps)) * m_w_hat, b - (self.lr / (np.sqrt(v_b_hat) + eps)) * m_b_hat

# 3. Training Logic
def train_perceptron(method, batch_size=None):
    w, b = np.zeros((2, 1)), 0
    opt = Optimizers(lr=0.1)
    history = []
    
    for epoch in range(50):
        # Handle Batching Logic
        if method == "BGD": # Batch
            indices = np.arange(len(X))
        elif method == "SGD": # Stochastic
            indices = [np.random.randint(len(X))]
        else: # Mini-Batch
            indices = np.random.choice(len(X), batch_size or 10, replace=False)
            
        x_batch, y_batch = X[indices], y[indices]
        
        # Forward Pass (Sigmoid Activation)
        z = np.dot(x_batch, w) + b
        y_pred = 1 / (1 + np.exp(-z))
        
        # Loss (MSE for simplicity)
        loss = np.mean((y_pred - y_batch)**2)
        history.append(loss)
        
        # Backward Pass
        dw = np.dot(x_batch.T, (y_pred - y_batch)) / len(indices)
        db = np.sum(y_pred - y_batch) / len(indices)
        
        # Update using selected Optimizer
        if method in ["BGD", "SGD", "MBGD"]: w, b = opt.gd(w, b, dw, db)
        elif method == "Momentum": w, b = opt.momentum(w, b, dw, db)
        elif method == "AdaGrad": w, b = opt.adagrad(w, b, dw, db)
        elif method == "RMSprop": w, b = opt.rmsprop(w, b, dw, db)
        elif method == "Adam": w, b = opt.adam(w, b, dw, db)
            
    return history

# 4. Compare Results
methods = ["BGD", "SGD", "MBGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]
plt.figure(figsize=(10, 5))
for m in methods:
    plt.plot(train_perceptron(m), label=m)

plt.title("Perceptron Convergence: Optimizer Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
