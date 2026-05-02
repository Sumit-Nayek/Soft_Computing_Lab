#______________
# a> Implement a simple CNN for the given image classification.
#______________
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Setup: Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load and preprocess MNIST dataset
# MNIST: 28x28 grayscale images of digits 0-9
# Convert to tensor and normalize to range [-1, 1] (mean=0.5, std=0.5)
transform = transforms.Compose([
    transforms.ToTensor(),                     # Convert PIL image to tensor (0-1 range)
    transforms.Normalize((0.5,), (0.5,))       # Normalize to (-1, 1) for better training
])

# Download MNIST (first run) - very reliable, small file size
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 3. Define a very simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolution: input 1 channel (grayscale), output 16 channels, kernel 3x3, padding 1 (keeps size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)   # reduces 28x28 -> 14x14
        
        # Second convolution: 16 channels -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)   # reduces 14x14 -> 7x7
        
        # After two pools, feature map size = 7x7 with 32 channels = 32*7*7 = 1568 numbers
        self.fc = nn.Linear(32 * 7 * 7, num_classes)   # final fully connected layer to 10 classes
    
    def forward(self, x):
        # Forward pass through layers
        x = self.pool1(self.relu1(self.conv1(x)))   # conv1 → relu → maxpool
        x = self.pool2(self.relu2(self.conv2(x)))   # conv2 → relu → maxpool
        x = x.view(x.size(0), -1)                  # flatten: (batch, 32*7*7)
        x = self.fc(x)                             # linear layer to get class scores (logits)
        return x

model = SimpleCNN(num_classes=10).to(device)
print(model)   # prints the architecture

# 4. Loss function and optimizer
criterion = nn.CrossEntropyLoss()   # combines softmax + negative log-likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)   # Adam is a popular optimizer

# 5. Train the network
num_epochs = 5   # small number for quick learning; can increase for better accuracy
print("Starting training...")
for epoch in range(num_epochs):
    model.train()          # set model to training mode (enables dropout/batchnorm if any)
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)   # move data to GPU/CPU
        
        # Forward pass: compute predictions
        outputs = model(images)
        loss = criterion(outputs, labels)   # calculate loss
        
        # Backward pass and optimization
        optimizer.zero_grad()   # clear previous gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights        
        running_loss += loss.item()   # accumulate loss    
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_loss:.4f}")

# 6. Evaluate on test set
model.eval()        # set to evaluation mode (no gradients, dropout off)
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():   # disable gradient computation (saves memory and speed)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)   # multiply by batch size for total loss
        
        _, predicted = torch.max(outputs, 1)   # get the class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= total   # average loss per sample
accuracy = 100.0 * correct / total

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
#_____________
# b> Write a program to optimize a function f(x) = ∑ xid 2i using Differential Evolution and make a comparative study with Genetic Algorithm and Simulated Annealing with the given domain range
#-------------
[-10,10] and global minima at Zero.
import numpy as np
from scipy.optimize import differential_evolution

def sphere(x):
    return np.sum(x**2)

bounds = [(-10, 10)] * 10
result_de = differential_evolution(sphere, bounds, strategy='best1bin', popsize=15, tol=1e-8, maxiter=200, seed=42)

print("=== Differential Evolution ===")
print("Best solution:", result_de.x)
print("Best value (f(x)):", result_de.fun)
print("Success:", result_de.success)

import random

def genetic_algorithm(objective, bounds, n_pop=100, n_gen=200, crossover_rate=0.8, mutation_rate=0.1, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    dim = len(bounds)
    
    # Initialize population
    population = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_pop, dim))
    
    best_solution = None
    best_eval = float('inf')
    history = []
    
    for gen in range(n_gen):
        # Evaluate
        fitness = np.array([objective(ind) for ind in population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_eval:
            best_eval = fitness[min_idx]
            best_solution = population[min_idx].copy()
        
        history.append(best_eval)
        
        # Selection (tournament)
        parents = []
        for _ in range(n_pop):
            idx1, idx2 = random.sample(range(n_pop), 2)
            winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
            parents.append(population[winner])
        parents = np.array(parents)
        
        # Crossover (arithmetic)
        offspring = []
        for i in range(0, n_pop, 2):
            if random.random() < crossover_rate and i+1 < n_pop:
                alpha = random.random()
                child1 = alpha * parents[i] + (1 - alpha) * parents[i+1]
                child2 = (1 - alpha) * parents[i] + alpha * parents[i+1]
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1] if i+1 < n_pop else parents[i]])
        
        population = np.array(offspring[:n_pop])
        
        # Mutation
        for ind in population:
            if random.random() < mutation_rate:
                idx = random.randint(0, dim-1)
                ind[idx] += np.random.normal(0, 1.0)  # Gaussian mutation
                ind = np.clip(ind, [b[0] for b in bounds], [b[1] for b in bounds])
    
    return best_solution, best_eval, history

best_ga, eval_ga, _ = genetic_algorithm(sphere, bounds, seed=42)
print("\n=== Genetic Algorithm ===")
print("Best solution:", best_ga)
print("Best value (f(x)):", eval_ga)

import math

def simulated_annealing(objective, bounds, n_iterations=10000, temp=1000.0, cooling_rate=0.995, step_size=0.5, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    dim = len(bounds)
    
    # Initial solution
    current = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], dim)
    current_eval = objective(current)
    best, best_eval = current.copy(), current_eval
    
    for i in range(n_iterations):
        # Generate neighbor
        candidate = current + np.random.normal(0, step_size, dim)
        candidate = np.clip(candidate, [b[0] for b in bounds], [b[1] for b in bounds])
        candidate_eval = objective(candidate)
        
        # Acceptance criterion
        if candidate_eval < current_eval or random.random() < math.exp((current_eval - candidate_eval) / (temp + 1e-8)):
            current, current_eval = candidate, candidate_eval
        
        if candidate_eval < best_eval:
            best, best_eval = candidate.copy(), candidate_eval
        
        temp *= cooling_rate  # Cool down
    
    return best, best_eval

best_sa, eval_sa = simulated_annealing(sphere, bounds, seed=42)
print("\n Simulated Annealing ")
print("Best solution:", best_sa)
print("Best value (f(x)):", eval_sa)

print("\n Comparative Results (10D Sphere Function) ")
print(f"Differential Evolution : f(x) = {result_de.fun:.10f}")
print(f"Genetic Algorithm      : f(x) = {eval_ga:.10f}")
print(f"Simulated Annealing    : f(x) = {eval_sa:.10f}")

