"""SeeDL Quickstart: Visualizing tensors, matrices, and images effortlessly."""

import seedl as sd

# 1. Visualize a scalar with custom unit / note
v1 = sd.see(0.0035, backend="svg")
print(f"Scalar view: {v1.summary}")

# 2. Visualize a 1D sequence or signal
v2 = sd.see([1.2, 2.5, 3.1, 2.8, 4.0, 5.2, 4.9], backend="svg")
print(f"Sequence view: {v2.summary}")

# 3. Visualize a 2D matrix (heatmap)
matrix = [
    [1.0, 0.5, 0.2],
    [0.5, 2.0, 0.8],
    [0.2, 0.8, 1.5],
]
v3 = sd.see(matrix, backend="svg")
print(f"Matrix view: {v3.summary}")

# 4. Export visualization to standalone self-contained HTML
# v3.save("correlation_matrix.html")
print("Quickstart completed successfully.")
