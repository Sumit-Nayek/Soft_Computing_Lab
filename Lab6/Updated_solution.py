## a>

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
