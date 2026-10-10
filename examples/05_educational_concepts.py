"""SeeDL Educational Simulations: Learn deep learning concepts visually."""

import seedl as sd

# 1. Gradient Descent Simulation
print("--- Gradient Descent ---")
gd_view = sd.learn("gradient_descent", lr=0.15, steps=10, backend="svg")
print(gd_view.summary)

# 2. Backpropagation Chain Rule Graph
print("\n--- Backpropagation ---")
bp_view = sd.learn("backpropagation", backend="svg")
print(bp_view.summary)

# 3. Transformer Scaled Dot-Product Attention
print("\n--- Self-Attention ---")
att_view = sd.learn("attention", backend="svg")
print(att_view.summary)

# 4. Activation Functions
print("\n--- Activation Functions ---")
act_view = sd.learn("activation_functions", backend="svg")
print(act_view.summary)

# 5. Convolution Operation Simulation
print("\n--- Convolution Simulation ---")
conv_view = sd.convolution(backend="svg")
print(conv_view.summary)
