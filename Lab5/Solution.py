
import numpy as np

# TRAINING PHASE

def train_perceptron():
    # Training data: [punctuality, task_completion, teamwork], label
    training_data = [
        ([1, 1, 1], 1),  # High Performer
        ([1, 1, 0], 1),  # High Performer
        ([1, 0, 1], 0),  # Low Performer
        ([0, 1, 1], 1),  # High Performer
        ([0, 0, 0], 0),  # Low Performer
        ([1, 0, 0], 0),  # Low Performer
        ([0, 1, 0], 0),  # Low Performer
        ([0, 0, 1], 0)   # Low Performer
    ]

    # Initialize weights and bias
    w = [0.0, 0.0, 0.0]
    b = 0.0
    learning_rate = 0.1
    epochs = 100


    print("TRAINING PHASE")


    # Training loop
    for epoch in range(epochs):
        total_errors = 0

        for features, actual in training_data:
            x1, x2, x3 = features

            # Calculate weighted sum
            z = (w[0]*x1) + (w[1]*x2) + (w[2]*x3) + b

            # Activation function
            predicted = 1 if z >= 0 else 0

            # Calculate error
            error = actual - predicted

            # Update if misclassified
            if error != 0:
                w[0] = w[0] + learning_rate * error * x1
                w[1] = w[1] + learning_rate * error * x2
                w[2] = w[2] + learning_rate * error * x3
                b = b + learning_rate * error
                total_errors += 1

        # Print progress
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Errors = {total_errors}, Weights = {w}, Bias = {b:.2f}")

        # Stop if all correct
        if total_errors == 0:
            print(f"\nConverged at Epoch {epoch}")
            break

    print(f"\nFinal Weights: w1={w[0]:.2f}, w2={w[1]:.2f}, w3={w[2]:.2f}")
    print(f"Final Bias: b={b:.2f}")


    return w, b

# PREDICTION PHASE

def predict_performance(punctuality, task_completion, teamwork, w, b):
    # Calculate weighted sum
    z = (w[0]*punctuality) + (w[1]*task_completion) + (w[2]*teamwork) + b

    # Apply activation function
    if z >= 0:
        return 1  # High Performer
    else:
        return 0  # Low Performer
# MAIN PROGRAM

def main():
    # Train the perceptron
    weights, bias = train_perceptron()

    # Test with new employees

    print("PREDICTION PHASE")


    test_employees = [
        {"name": "Alice", "punctuality": 1, "task_completion": 1, "teamwork": 1},
        {"name": "Bob", "punctuality": 1, "task_completion": 0, "teamwork": 0},
        {"name": "Charlie", "punctuality": 0, "task_completion": 1, "teamwork": 1},
        {"name": "David", "punctuality": 0, "task_completion": 0, "teamwork": 0},
        {"name": "Emma", "punctuality": 1, "task_completion": 1, "teamwork": 0}
    ]

    for emp in test_employees:
        result = predict_performance(
            emp["punctuality"],
            emp["task_completion"],
            emp["teamwork"],
            weights,
            bias
        )

        performance = "High Performer" if result == 1 else "Low Performer"

        print(f"\nEmployee: {emp['name']}")
        print(f"  Punctuality: {emp['punctuality']}")
        print(f"  Task Completion: {emp['task_completion']}")
        print(f"  Teamwork: {emp['teamwork']}")
        print(f"  Result: {performance}")

    # Calculate weighted sum for each

    print("DETAILED CALCULATIONS")


    for emp in test_employees:
        z = (weights[0]*emp["punctuality"]) + (weights[1]*emp["task_completion"]) + \
            (weights[2]*emp["teamwork"]) + bias
        print(f"{emp['name']}: z = {z:.2f} → {'High' if z>=0 else 'Low'} Performer")

# RUN THE PROGRAM
if __name__ == "__main__":
    main()