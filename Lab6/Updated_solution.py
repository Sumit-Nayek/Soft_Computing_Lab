#-----------
## a>
#-----------
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

#--------
# b>
# -------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# MLP implementation (Adam)

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        # He initialization
        self.W = {}
        self.b = {}
        for i in range(1, len(layer_sizes)):
            self.W[i] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0/layer_sizes[i-1])
            self.b[i] = np.zeros((1, layer_sizes[i]))

        # Adam accumulators
        self.m_W = {i: np.zeros_like(self.W[i]) for i in range(1, len(layer_sizes))}
        self.v_W = {i: np.zeros_like(self.W[i]) for i in range(1, len(layer_sizes))}
        self.m_b = {i: np.zeros_like(self.b[i]) for i in range(1, len(layer_sizes))}
        self.v_b = {i: np.zeros_like(self.b[i]) for i in range(1, len(layer_sizes))}
        self.t = 0

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, len(self.layer_sizes)):
            Z = A @ self.W[i] + self.b[i]
            if i == len(self.layer_sizes)-1:   # output layer
                A = self._softmax(Z)
            else:                               # hidden layer
                A = self._relu(Z)
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
        return A, cache

    def backward(self, X, y_onehot, cache):
        m = X.shape[0]
        grads_W = {}
        grads_b = {}

        # Output layer gradient (softmax + cross-entropy)
        L = len(self.layer_sizes)-1
        dZ = cache[f'A{L}'] - y_onehot

        for i in range(L, 0, -1):
            A_prev = cache[f'A{i-1}']
            grads_W[i] = (A_prev.T @ dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 1:
                dA = dZ @ self.W[i].T
                dZ = dA * (cache[f'Z{i-1}'] > 0)   # ReLU derivative

        return grads_W, grads_b

    def update_adam(self, grads_W, grads_b):
        self.t += 1
        for i in range(1, len(self.layer_sizes)):
            # Weights
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (grads_W[i]**2)
            m_hat = self.m_W[i] / (1 - self.beta1**self.t)
            v_hat = self.v_W[i] / (1 - self.beta2**self.t)
            self.W[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i]**2)
            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)
            self.b[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    def train(self, X, y, epochs=200, batch_size=32, verbose=False):
        n_samples = X.shape[0]
        n_classes = self.layer_sizes[-1]
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                y_onehot = np.eye(n_classes)[y_batch]

                probs, cache = self.forward(X_batch)
                loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
                epoch_loss += loss * len(X_batch)

                grads_W, grads_b = self.backward(X_batch, y_onehot, cache)
                self.update_adam(grads_W, grads_b)

            epoch_loss /= n_samples
            if verbose and (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

# Load and process datasets
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    # Convert labels to 0-based indexing
    y = y - np.min(y)
    return X, y

datasets = {
    "d1.txt": (load_dataset("d1.txt"), "3 classes"),
    "d2.TXT": (load_dataset("d2.TXT"), "5 classes"),
    "d3.txt": (load_dataset("d3.txt"), "9 classes")
}

results = {}
for fname, ((X, y), desc) in datasets.items():
    print(f"\n=== {fname} ({desc}) ===")
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = len(np.unique(y))
    # MLP architecture: input_dim=2, hidden1=64, hidden2=32, output=n_classes
    mlp = MLP(layer_sizes=[2, 64, 32, n_classes], learning_rate=0.001)

    # Train
    mlp.train(X_train, y_train, epochs=300, batch_size=32, verbose=True)

    # Evaluate
    train_acc = mlp.score(X_train, y_train)
    test_acc = mlp.score(X_test, y_test)

    # Compute test loss
    probs_test, _ = mlp.forward(X_test)
    test_loss = -np.mean(np.log(probs_test[np.arange(len(y_test)), y_test] + 1e-8))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")
    results[fname] = (test_loss, test_acc)
## b>
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(8, activation='sigmoid', input_shape=(4,)),   # Hidden layer
    keras.layers.Dense(3, activation='sigmoid')                     # Output layer
])

model.compile(
    optimizer='adam',                    # You can change to 'sgd', 'rmsprop'
    loss='mean_squared_error',           # MSE loss (as used in previous codes)
    metrics=['accuracy']
)

print("Training MLP with TensorFlow...\n")
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Optional: Show final predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(f"Final Accuracy: {np.mean(y_pred == y_true)*100:.2f}%")
