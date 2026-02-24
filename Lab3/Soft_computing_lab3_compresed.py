"""
Compressed Fuzzy Disaster Risk Assessment System
Using only functions and basic programming (No classes)
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# PART 1: DATA INITIALIZATION

# Define sets
hazards = {'H1': 'Earthquake', 'H2': 'Flood', 'H3': 'Cyclone', 'H4': 'Landslide', 'H5': 'Drought'}
vulnerabilities = {'V1': 'Population Density', 'V2': 'Infrastructure Quality', 
                   'V3': 'Emergency Response Capacity', 'V4': 'Economic Resources', 
                   'V5': 'Geographic Location'}
risk_levels = {'R1': 'Low Risk', 'R2': 'Moderate Risk', 'R3': 'High Risk', 'R4': 'Critical Risk'}

hazard_list = list(hazards.keys())
vulnerability_list = list(vulnerabilities.keys())
risk_list = list(risk_levels.keys())

# H × V Relation Matrix (Hazards → Vulnerabilities)
H_V = np.array([
    [0.9, 0.8, 0.7, 0.6, 0.8],  # H1: Earthquake
    [0.6, 0.9, 0.5, 0.4, 0.9],  # H2: Flood
    [0.8, 0.7, 0.6, 0.3, 0.7],  # H3: Cyclone
    [0.4, 0.6, 0.3, 0.2, 0.9],  # H4: Landslide
    [0.3, 0.2, 0.4, 0.9, 0.5]   # H5: Drought
])

# V × R Relation Matrix (Vulnerabilities → Risk Levels)
V_R = np.array([
    [0.2, 0.4, 0.8, 0.9],  # V1: Population Density
    [0.8, 0.7, 0.3, 0.1],  # V2: Infrastructure Quality
    [0.9, 0.8, 0.4, 0.2],  # V3: Emergency Response
    [0.7, 0.6, 0.3, 0.2],  # V4: Economic Resources
    [0.1, 0.3, 0.6, 0.9]   # V5: Geographic Location
])

# PART 2: DISPLAY FUNCTIONS


def display_matrices():
    """Display all fuzzy relation matrices"""

    print("FUZZY RELATION MATRICES")

    
    # Display H×V Matrix
    print("\n1. HAZARD-VULNERABILITY RELATION (H × V):")
    print("     ", end="")
    for v in vulnerability_list:
        print(f"{v:>6}", end="")
    print("\n     " + "-" * 35)
    for i, h in enumerate(hazard_list):
        print(f"{h}  |", end="")
        for j in range(len(vulnerability_list)):
            print(f"{H_V[i, j]:6.2f}", end="")
        print(f"  |  {hazards[h]}")
    
    # Display V×R Matrix
    print("\n2. VULNERABILITY-RISK RELATION (V × R):")
    print("     ", end="")
    for r in risk_list:
        print(f"{r:>6}", end="")
    print("\n     " + "-" * 30)
    for i, v in enumerate(vulnerability_list):
        print(f"{v}  |", end="")
        for j in range(len(risk_list)):
            print(f"{V_R[i, j]:6.2f}", end="")
        print(f"  |  {vulnerabilities[v]}")

# PART 3: COMPOSITION FUNCTIONS


def max_min_composition(A, B):
    """Max-Min composition"""
    m, n = A.shape
    n, p = B.shape
    result = np.zeros((m, p))
    for i in range(m):
        for k in range(p):
            result[i, k] = max([min(A[i, j], B[j, k]) for j in range(n)])
    return result

def max_product_composition(A, B):
    """Max-Product composition"""
    m, n = A.shape
    n, p = B.shape
    result = np.zeros((m, p))
    for i in range(m):
        for k in range(p):
            result[i, k] = max([A[i, j] * B[j, k] for j in range(n)])
    return result

# Calculate compositions
max_min_result = max_min_composition(H_V, V_R)
max_product_result = max_product_composition(H_V, V_R)

def display_compositions():
    """Display composition results"""

    print("FUZZY COMPOSITION RESULTS")
 
    
    print("\nMAX-MIN COMPOSITION (Standard Risk):")
    print("     ", end="")
    for r in risk_list:
        print(f"{r:>8}", end="")
    print("\n     " + "-" * 40)
    for i, h in enumerate(hazard_list):
        print(f"{h}  |", end="")
        for j in range(len(risk_list)):
            print(f"{max_min_result[i, j]:8.3f}", end="")
        print(f"  |  {hazards[h]}")
    
    print("\nMAX-PRODUCT COMPOSITION (Intensified Risk):")
    print("     ", end="")
    for r in risk_list:
        print(f"{r:>8}", end="")
    print("\n     " + "-" * 40)
    for i, h in enumerate(hazard_list):
        print(f"{h}  |", end="")
        for j in range(len(risk_list)):
            print(f"{max_product_result[i, j]:8.3f}", end="")
        print(f"  |  {hazards[h]}")

# PART 4: ANALYSIS FUNCTIONS

def identify_critical_risks(threshold=0.5):
    """Identify critical risk scenarios"""
    critical = []
    for i, h in enumerate(hazard_list):
        for j, r in enumerate(risk_list):
            val = max(max_min_result[i, j], max_product_result[i, j])
            if val > threshold:
                critical.append({
                    'hazard': hazards[h],
                    'risk': risk_levels[r],
                    'value': val,
                    'priority': 'HIGH' if val > 0.7 else 'MEDIUM'
                })
    return sorted(critical, key=lambda x: x['value'], reverse=True)

# PART 5: VISUALIZATION FUNCTIONS


def plot_heatmaps():
    """Create simple heatmaps"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    matrices = [H_V, V_R, max_min_result, max_product_result]
    titles = ["Hazard-Vulnerability", "Vulnerability-Risk", 
              "Max-Min Composition", "Max-Product Composition"]
    xlabels = [vulnerability_list, risk_list, risk_list, risk_list]
    ylabels = [hazard_list, vulnerability_list, hazard_list, hazard_list]
    
    for idx, (ax, mat, title, xl, yl) in enumerate(zip(axes.flat, matrices, titles, xlabels, ylabels)):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(xl)))
        ax.set_yticks(range(len(yl)))
        ax.set_xticklabels(xl, rotation=45, ha='right')
        ax.set_yticklabels(yl)
        ax.set_title(title)
        
        for i in range(len(yl)):
            for j in range(len(xl)):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Fuzzy Disaster Risk Relations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comparison():
    """Compare Max-Min vs Max-Product with values on top of bars"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(hazard_list))
    width = 0.35
    
    mm_avg = np.mean(max_min_result, axis=1)
    mp_avg = np.mean(max_product_result, axis=1)
    
    bars1 = ax.bar(x - width/2, mm_avg, width, label='Max-Min', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, mp_avg, width, label='Max-Product', color='salmon', edgecolor='black')
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Hazards', fontsize=12)
    ax.set_ylabel('Average Risk Level', fontsize=12)
    ax.set_title('Risk Assessment Comparison (Values on Top)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([hazards[h] for h in hazard_list], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()

# PART 6: MAIN EXECUTION


# Display matrices
display_matrices()

# Display compositions
display_compositions()

# Display critical risks

print("CRITICAL RISK SCENARIOS")

critical = identify_critical_risks(0.5)
for item in critical[:5]:
    print(f"  • {item['hazard']} - {item['risk']}: {item['value']:.3f} ({item['priority']})")


# Generate visualizations
print("\nGenerating visualizations...")
plot_heatmaps()
plot_comparison()
