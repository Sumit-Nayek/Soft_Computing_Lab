## Comparing the value of Optimizer in the backpropagation of the neural network with respect to single neuron
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 2)  # 200 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # Simple classification

# Neural network parameters - PERCEPTRON (Single Neuron)
input_size = 2
hidden_size = 0  # No hidden layer
output_size = 1  # Single output neuron
learning_rate = 0.1
epochs = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass_perceptron(X, W, b):
    """Forward pass for single neuron (perceptron)"""
    z = np.dot(X, W) + b
    a = sigmoid(z)
    return z, a

def compute_loss(y_true, y_pred):
    """Binary cross-entropy loss"""
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

def backward_pass_perceptron(X, y, z, a):
    """Backward pass for single neuron"""
    m = X.shape[0]

    # Gradient computation
    dz = a - y
    dW = np.dot(X.T, dz) / m
    db = np.sum(dz, axis=0, keepdims=True) / m

    return dW, db

def train_perceptron(optimizer_type, X, y, learning_rate, epochs, batch_size=32):
    """Train single neuron perceptron with specified optimizer"""

    # Initialize weights and bias
    W = np.random.randn(input_size, output_size) * 0.5
    b = np.zeros((1, output_size))

    # Storage for tracking
    losses = []
    accuracies = []
    weight_updates = []
    weights_history = []
    decision_boundary_history = []  # Store decision boundary parameters

    # Initial weights snapshot
    weights_history.append(W.copy())

    for epoch in range(epochs):
        if optimizer_type == 'gd':
            # Gradient Descent - full batch
            z, a = forward_pass_perceptron(X, W, b)
            loss = compute_loss(y, a)
            dW, db = backward_pass_perceptron(X, y, z, a)

            W -= learning_rate * dW
            b -= learning_rate * db

            update_magnitude = np.mean(np.abs(dW))

        elif optimizer_type == 'bgd':
            # Batch Gradient Descent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                z, a = forward_pass_perceptron(X_batch, W, b)
                batch_loss = compute_loss(y_batch, a)
                epoch_loss += batch_loss * len(X_batch)
                dW, db = backward_pass_perceptron(X_batch, y_batch, z, a)

                W -= learning_rate * dW
                b -= learning_rate * db

            loss = epoch_loss / X.shape[0]
            update_magnitude = np.mean(np.abs(dW))

        elif optimizer_type == 'sgd':
            # Stochastic Gradient Descent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            update_magnitude = 0

            for i in range(X.shape[0]):
                X_sample = X_shuffled[i:i+1]
                y_sample = y_shuffled[i:i+1]

                z, a = forward_pass_perceptron(X_sample, W, b)
                sample_loss = compute_loss(y_sample, a)
                epoch_loss += sample_loss
                dW, db = backward_pass_perceptron(X_sample, y_sample, z, a)

                W -= learning_rate * dW
                b -= learning_rate * db

                update_magnitude += np.mean(np.abs(dW))

            loss = epoch_loss / X.shape[0]
            update_magnitude /= X.shape[0]

        losses.append(loss)
        weight_updates.append(update_magnitude)

        # Calculate accuracy
        _, predictions = forward_pass_perceptron(X, W, b)
        accuracy = np.mean((predictions > 0.5).astype(int) == y)
        accuracies.append(accuracy)

        # Store weights periodically
        if epoch % 20 == 0:
            weights_history.append(W.copy())
            decision_boundary_history.append((W.copy(), b.copy()))

    return losses, accuracies, weight_updates, W, b, weights_history, decision_boundary_history

# Train all optimizers on perceptron
print("Training Single Neuron Perceptron...")
print("="*60)

gd_losses, gd_accuracies, gd_updates, W_gd, b_gd, gd_weights, gd_boundaries = train_perceptron('gd', X, y, learning_rate, epochs)
bgd_losses, bgd_accuracies, bgd_updates, W_bgd, b_bgd, bgd_weights, bgd_boundaries = train_perceptron('bgd', X, y, learning_rate, epochs, batch_size=32)
sgd_losses, sgd_accuracies, sgd_updates, W_sgd, b_sgd, sgd_weights, sgd_boundaries = train_perceptron('sgd', X, y, learning_rate, epochs)

# ======== VISUALIZATION 1: Perceptron Architecture ========

fig = plt.figure(figsize=(20, 14))

# 1.1 Perceptron Architecture
ax1 = plt.subplot(3, 3, 1)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 4)
ax1.axis('off')

# Draw input layer
for i in range(input_size):
    circle = plt.Circle((1, 1.5 + i*0.8), 0.2, fill=True, color='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(circle)
ax1.text(1, 0.8, 'Input Layer\n(2 features)', ha='center', fontsize=10, fontweight='bold')

# Draw output neuron
circle = plt.Circle((4, 1.9), 0.3, fill=True, color='lightcoral', edgecolor='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(4, 1.1, 'Output Neuron\n(1 neuron)', ha='center', fontsize=10, fontweight='bold')
ax1.text(4, 2.3, 'σ(z)', ha='center', fontsize=8, fontstyle='italic')

# Draw connections
for i in range(input_size):
    ax1.plot([1.2, 3.7], [1.5 + i*0.8, 1.9], 'gray', linewidth=1.5, alpha=0.7)

# Add bias
ax1.annotate('bias', xy=(2.5, 1.9), xytext=(2.5, 2.8),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
ax1.text(2.5, 2.9, 'b', ha='center', fontsize=10)

ax1.set_title('Perceptron Architecture\n(Single Neuron)', fontsize=12, fontweight='bold')

# 1.2 Perceptron Decision Function
ax2 = plt.subplot(3, 3, 2)
ax2.axis('off')
perceptron_eq = """Perceptron Decision Function:

z = w₁x₁ + w₂x₂ + b

ŷ = σ(z) = 1 / (1 + e⁻ᶻ)

Decision Boundary:
w₁x₁ + w₂x₂ + b = 0

Where:
• x₁, x₂: Input features
• w₁, w₂: Weights
• b: Bias
• ŷ: Output probability
• σ: Sigmoid activation

Classification:
ŷ > 0.5 → Class 1
ŷ ≤ 0.5 → Class 0"""
ax2.text(0.1, 0.95, perceptron_eq, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace')
ax2.set_title('Perceptron Mathematical Model', fontsize=12, fontweight='bold')


# ======= VISUALIZATION 2: Training Dynamics ======
# 2.1 Loss Curves
plt.subplot(3, 3, 3)
plt.plot(gd_losses, label='GD', linewidth=2, alpha=0.8)
plt.plot(bgd_losses, label='BGD', linewidth=2, alpha=0.8)
plt.plot(sgd_losses, label='SGD', linewidth=2, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves\n(Perceptron)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.2 Accuracy Curves
plt.subplot(3, 3, 4)
plt.plot(gd_accuracies, label='GD', linewidth=2, alpha=0.8)
plt.plot(bgd_accuracies, label='BGD', linewidth=2, alpha=0.8)
plt.plot(sgd_accuracies, label='SGD', linewidth=2, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curves\n(Perceptron)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# 2.3 Weight Update Magnitudes
plt.subplot(3, 3, 5)
plt.plot(gd_updates, label='GD', alpha=0.7)
plt.plot(bgd_updates, label='BGD', alpha=0.7)
plt.plot(sgd_updates, label='SGD', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Update Magnitude')
plt.title('Weight Update Magnitudes')
plt.legend()
plt.grid(True, alpha=0.3)

# 2.4 Final Performance Comparison
plt.subplot(3, 3, 6)
x = np.arange(3)
width = 0.35
final_acc = [gd_accuracies[-1], bgd_accuracies[-1], sgd_accuracies[-1]]
final_loss = [gd_losses[-1], bgd_losses[-1], sgd_losses[-1]]

bars1 = plt.bar(x - width/2, final_acc, width, label='Accuracy', color='green', alpha=0.7)
bars2 = plt.bar(x + width/2, final_loss, width, label='Loss', color='red', alpha=0.7)
plt.xlabel('Optimizer')
plt.ylabel('Score')
plt.title('Final Performance Comparison')
plt.xticks(x, ['GD', 'BGD', 'SGD'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# ==================== VISUALIZATION 3: Weight Evolution ====================
# 3.1 Weight Evolution (GD)
ax9 = plt.subplot(3, 3, 7)
weights_gd_array = np.array([w.flatten() for w in gd_weights])
for i in range(weights_gd_array.shape[1]):
    ax9.plot(weights_gd_array[:, i], label=f'w{i+1}', linewidth=2)
ax9.set_xlabel('Training Checkpoint (×20 epochs)')
ax9.set_ylabel('Weight Value')
ax9.set_title('Weight Evolution (GD)')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 3.2 Weight Evolution (BGD)
ax10 = plt.subplot(3, 3, 8)
weights_bgd_array = np.array([w.flatten() for w in bgd_weights])
for i in range(weights_bgd_array.shape[1]):
    ax10.plot(weights_bgd_array[:, i], label=f'w{i+1}', linewidth=2)
ax10.set_xlabel('Training Checkpoint (×20 epochs)')
ax10.set_ylabel('Weight Value')
ax10.set_title('Weight Evolution (BGD)')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 3.3 Weight Evolution (SGD)
ax11 = plt.subplot(3, 3, 9)
weights_sgd_array = np.array([w.flatten() for w in sgd_weights])
for i in range(weights_sgd_array.shape[1]):
    ax11.plot(weights_sgd_array[:, i], label=f'w{i+1}', linewidth=2)
ax11.set_xlabel('Training Checkpoint (×20 epochs)')
ax11.set_ylabel('Weight Value')
ax11.set_title('Weight Evolution (SGD)')
ax11.legend()
ax11.grid(True, alpha=0.3)

# ======= VISUALIZATION 6: Loss Landscape ======

# 6.1 3D Loss Landscape
ax1 = fig4.add_subplot(131, projection='3d')
w1_range = np.linspace(-2, 2, 50)
w2_range = np.linspace(-2, 2, 50)
W1_mesh, W2_mesh = np.meshgrid(w1_range, w2_range)
loss_mesh = np.zeros_like(W1_mesh)

for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        temp_W = np.array([[W1_mesh[i, j]], [W2_mesh[i, j]]])
        _, pred = forward_pass_perceptron(X, temp_W, b_gd)
        loss_mesh[i, j] = compute_loss(y, pred)

surf = ax1.plot_surface(W1_mesh, W2_mesh, loss_mesh, cmap='viridis', alpha=0.7)
ax1.set_xlabel('w₁')
ax1.set_ylabel('w₂')
ax1.set_zlabel('Loss')
ax1.set_title('Loss Landscape\n(2D Weight Space)')
plt.colorbar(surf, ax=ax1, fraction=0.05)

# 6.2 Contour Plot with Optimization Path
ax2 = fig4.add_subplot(132)
contour = ax2.contourf(W1_mesh, W2_mesh, loss_mesh, levels=20, cmap='viridis', alpha=0.8)
ax2.set_xlabel('w₁')
ax2.set_ylabel('w₂')
ax2.set_title('Loss Contour with Optimization Path')

# Plot optimization path for GD
weights_path = np.array([w.flatten() for w in gd_weights])
ax2.plot(weights_path[:, 0], weights_path[:, 1], 'r.-', linewidth=2, markersize=8, label='GD Path')
ax2.plot(weights_path[0, 0], weights_path[0, 1], 'go', markersize=10, label='Start')
ax2.plot(weights_path[-1, 0], weights_path[-1, 1], 'ro', markersize=10, label='End')
plt.colorbar(contour, ax=ax2)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 Learning Rate Effect
ax3 = fig4.add_subplot(133)
lr_values = [0.01, 0.05, 0.1, 0.5, 1.0]
losses_lr = []

for lr in lr_values:
    losses, _, _, _, _, _, _ = train_perceptron('gd', X, y, lr, 50)
    losses_lr.append(losses[-1])

ax3.plot(lr_values, losses_lr, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Learning Rate')
ax3.set_ylabel('Final Loss')
ax3.set_title('Effect of Learning Rate on Performance')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

plt.suptitle('Perceptron - Loss Landscape and Hyperparameter Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =========== VISUALIZATION 7: Gradient Analysis ======
fig5, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gradient magnitudes during training
axes[0, 0].plot(gd_updates, label='GD', linewidth=2)
axes[0, 0].plot(bgd_updates, label='BGD', linewidth=2)
axes[0, 0].plot(sgd_updates, label='SGD', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Gradient Magnitude')
axes[0, 0].set_title('Gradient Magnitudes Over Time (Perceptron)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss reduction efficiency
axes[0, 1].plot(np.cumsum(gd_updates), gd_losses, 'o-', label='GD', markersize=4)
axes[0, 1].plot(np.cumsum(bgd_updates), bgd_losses, 's-', label='BGD', markersize=4)
axes[0, 1].plot(np.cumsum(sgd_updates), sgd_losses, '^-', label='SGD', markersize=4)
axes[0, 1].set_xlabel('Cumulative Gradient Magnitude')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss vs Cumulative Gradient')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Learning curves with uncertainty
axes[1, 1].plot(gd_accuracies, label='GD', linewidth=2)
axes[1, 1].fill_between(range(epochs),
                        gd_accuracies - np.std(gd_accuracies)/2,
                        gd_accuracies + np.std(gd_accuracies)/2, alpha=0.2)
axes[1, 1].plot(bgd_accuracies, label='BGD', linewidth=2)
axes[1, 1].fill_between(range(epochs),
                        bgd_accuracies - np.std(bgd_accuracies)/2,
                        bgd_accuracies + np.std(bgd_accuracies)/2, alpha=0.2)
axes[1, 1].plot(sgd_accuracies, label='SGD', linewidth=2)
axes[1, 1].fill_between(range(epochs),
                        sgd_accuracies - np.std(sgd_accuracies)/2,
                        sgd_accuracies + np.std(sgd_accuracies)/2, alpha=0.2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Accuracy with Uncertainty Band')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Output activation distribution
axes[1, 0].hist(gd_weights[-1][0] * X[:, 0] + gd_weights[-1][1] * X[:, 1] + b_gd[0, 0],
                bins=20, alpha=0.5, label='GD', density=True)
axes[1, 0].hist(bgd_weights[-1][0] * X[:, 0] + bgd_weights[-1][1] * X[:, 1] + b_bgd[0, 0],
                bins=20, alpha=0.5, label='BGD', density=True)
axes[1, 0].hist(sgd_weights[-1][0] * X[:, 0] + sgd_weights[-1][1] * X[:, 1] + b_sgd[0, 0],
                bins=20, alpha=0.5, label='SGD', density=True)
axes[1, 0].set_xlabel('Pre-activation (z)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Output Pre-activation Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


# ========== VISUALIZATION 8: Training Progression ======
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))

# Plot training progression with checkpoints
epoch_samples = [0, 24, 49, 74, 99]

for idx, (optimizer_name, losses, accuracies) in enumerate([
    ('GD', gd_losses, gd_accuracies),
    ('BGD', bgd_losses, bgd_accuracies),
    ('SGD', sgd_losses, sgd_accuracies)
]):
    ax = axes6[idx]

    # Plot loss curve
    ax.plot(losses, 'b-', label='Loss', linewidth=1, alpha=0.7)
    ax_twin = ax.twinx()
    ax_twin.plot(accuracies, 'r-', label='Accuracy', linewidth=1, alpha=0.7)

    # Mark progression points
    for epoch in epoch_samples:
        if epoch < len(losses):
            ax.plot(epoch, losses[epoch], 'bo', markersize=6)
            ax_twin.plot(epoch, accuracies[epoch], 'ro', markersize=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='b')
    ax_twin.set_ylabel('Accuracy', color='r')
    ax.set_title(f'{optimizer_name} Training Progression\n(Perceptron)')
    ax.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.suptitle('Perceptron Training Progression with Marked Checkpoints', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

