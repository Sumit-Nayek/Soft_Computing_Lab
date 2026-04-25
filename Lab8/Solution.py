# a>
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print("Training finished!")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test images: {100 * correct / total:.2f}%')

# b>
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

