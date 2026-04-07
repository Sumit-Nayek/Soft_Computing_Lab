import numpy as np

# --- Step 1: Pairwise matrix (AHP for criteria weights) ---
M = np.array([
    [1,   3,   5,   7],
    [1/3, 1,   3,   5],
    [1/5, 1/3, 1,   3],
    [1/7, 1/5, 1/3, 1]
])

# Normalize and get weights
M_norm = M / M.sum(axis=0)
w = M_norm.mean(axis=1)

# --- Step 2: Decision matrix (Alternatives vs Criteria) ---
# Rows: Solar, Wind, Hydro, Biomass
# Columns: Cost, Efficiency, Env Impact, Reliability
X = np.array([
    [7, 8, 9, 6],   # Solar
    [6, 7, 8, 7],   # Wind
    [5, 9, 7, 8],   # Hydro
    [8, 6, 6, 5]    # Biomass
])

# --- Step 3: Normalize (TOPSIS) ---
R = X / np.sqrt((X**2).sum(axis=0))

# --- Step 4: Weighted matrix ---
V = R * w

# --- Step 5: Ideal best & worst ---
ideal_best = np.array([V[:,0].min(), V[:,1].max(), V[:,2].min(), V[:,3].max()])
ideal_worst = np.array([V[:,0].max(), V[:,1].min(), V[:,2].max(), V[:,3].min()])

# --- Step 6: Distances ---
D_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
D_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))

# --- Step 7: Closeness coefficient ---
CC = D_minus / (D_plus + D_minus)

# --- Result ---
alternatives = ["Solar", "Wind", "Hydro", "Biomass"]
ranking = sorted(zip(alternatives, CC), key=lambda x: x[1], reverse=True)

print("Ranking (Best to Worst):")
for i in ranking:
    print(i)