# test_mygrad.py

import mygrad

# Sample input array (e.g., y = x^2)
f = [0.0, 1.0, 4.0, 9.0, 16.0]  # Corresponds to x = 0, 1, 2, 3, 4
dx = 1.0
type = 1  # Example type value; depends on how you implemented it in C+
grad = mygrad.gradient_1d_order6(f)
print("Gradient:", grad)
