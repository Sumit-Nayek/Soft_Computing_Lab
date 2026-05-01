import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ====================== 1. Load and Prepare Data ======================
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================== 2. Activation Functions ======================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# ====================== 3. Forward Propagation ======================
def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# ====================== 4. Backpropagation & Weight Update ======================
def backprop(X, y, a1, a2, w2, lr):
    # Output layer error
    dz2 = a2 - y
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Hidden layer error
    dz1 = (dz2 @ w2.T) * sigmoid_derivative(a1)
    dw1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Update weights
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    
    return w1, b1, w2, b2

# ====================== 5. Training Loop ======================
def train(X_train, y_train, hidden_size=8, epochs=2000, lr=0.2):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    # Initialize weights and biases
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    
    for epoch in range(epochs):
        # Forward
        z1, a1, z2, a2 = forward(X_train, w1, b1, w2, b2)
        
        # Backpropagation + Update
        w1, b1, w2, b2 = backprop(X_train, y_train, a1, a2, w2, lr)
        
        if epoch % 400 == 0:
            loss = np.mean((a2 - y_train) ** 2)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
    
    return w1, b1, w2, b2

# ====================== 6. Prediction & Evaluation ======================
def predict(X, w1, b1, w2, b2):
    _, _, _, a2 = forward(X, w1, b1, w2, b2)
    return np.argmax(a2, axis=1)

# ====================== 7. Run Training ======================
print("Training MLP...\n")
w1, b1, w2, b2 = train(X_train, y_train, hidden_size=8, epochs=2000, lr=0.2)

# ====================== 8. Final Results ======================
y_pred = predict(X_test, w1, b1, w2, b2)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred == y_true) * 100
test_loss = np.mean((forward(X_test, w1, b1, w2, b2)[3] - y_test) ** 2)

print(f"Test Loss  : {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
