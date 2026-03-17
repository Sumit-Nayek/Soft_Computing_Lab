### Gredient Descent Working Principal
import matplotlib.pyplot as plt

# Gradient Function 
def compute_gradient(theta):
    return 2 * theta   # derivative of θ²


# Gradient Descent Core 
def run_gradient_descent(grad_func, initial_theta, learning_rate, iterations):
    theta = initial_theta
    
    history = {
        "iteration": [],
        "theta": [],
        "gradient": [],
        "update": [],
        "new_theta": []
    }
    
    for i in range(iterations):
        grad = grad_func(theta)
        update = learning_rate * grad
        theta_new = theta - update
        
        # Store everything
        history["iteration"].append(i)
        history["theta"].append(theta)
        history["gradient"].append(grad)
        history["update"].append(update)
        history["new_theta"].append(theta_new)
        
        theta = theta_new

    return theta, history


# Pretty Printing 
def print_results(history):
    print(f"{'Iteration':<10} | {'Theta (θt)':<12} | {'Gradient':<12} | {'Update':<12} | {'New Theta':<12}")

    
    for i in range(len(history["iteration"])):
        print(f"{history['iteration'][i]:<10} | "
              f"{history['theta'][i]:<12.4f} | "
              f"{history['gradient'][i]:<12.4f} | "
              f"{history['update'][i]:<12.4f} | "
              f"{history['new_theta'][i]:<12.4f}")


# Plot Function 
def plot_history(history):
    plt.figure()
    
    # Theta
    plt.plot(history["iteration"], history["theta"], marker='o', label="Theta (θ)")
    
    # Gradient
    plt.plot(history["iteration"], history["gradient"], marker='s', linestyle='--', label="Gradient")
    
    # Update
    plt.plot(history["iteration"], history["update"], marker='^', linestyle=':', label="Update (η·grad)")
    
    plt.xlabel("Iteration")
    plt.ylabel("Values")
    plt.title("Gradient Descent Parameter Updates")
    plt.legend()
    plt.grid()
    
    plt.show()


# MAIN 
initial_theta = 2.0
learning_rate = 0.1
iterations = 10

final_theta, history = run_gradient_descent(
    compute_gradient,
    initial_theta,
    learning_rate,
    iterations
)

# Print at the end
print_results(history)

print(f"\nFinal optimized theta: {final_theta:.4f}")

# Plot
plot_history(history)
