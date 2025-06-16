import numpy as np
import sys
import os

# Import the mygrad module from the C++ build directory
sys.path.insert(0, os.path.abspath("../build"))
import mygrad

def test_1d_gradient():
    print("=== 1D Gradient Test ===")
    f = [0.0, 1.0, 4.0, 9.0, 16.0]  # y = x^2
    dx = 1.0

    # Your gradient (C++ implementation)
    grad_mygrad = mygrad.gradient_1d_order6(f, dx)

    # NumPy gradient
    grad_numpy = np.gradient(f, dx, edge_order=2)

    print("Input f(x):", f)
    print("MyGrad Gradient:", grad_mygrad)
    print("NumPy Gradient:", grad_numpy)

    grad_mygrad_arr = np.array(grad_mygrad)
    diff = np.abs(grad_mygrad_arr - grad_numpy)
    print("Full Difference:", diff)
    print("Interior Difference (ignores edges):", diff[1:-1])


def test_2d_gradient_debug():
    print("\n=== 2D Gradient Debug Test ===")
    
    # Start with smaller array to debug
    print("Testing with 3x3 array first...")
    try:
        f_small = [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0], 
                   [7.0, 8.0, 9.0]]
        dx = 1.0
        dy = 1.0
        
        print("Calling gradient_2d_order2 with 3x3...")
        grad_flat = mygrad.gradient_2d_order2(f_small, dx, dy)
        print("Success! Got result:", len(grad_flat), "elements")
        
        print("Calling gradient_2d_order4 with 3x3...")
        grad_flat = mygrad.gradient_2d_order4(f_small, dx, dy)
        print("Success! Got result:", len(grad_flat), "elements")
        
        print("Calling gradient_2d_order6 with 3x3...")
        grad_flat = mygrad.gradient_2d_order6(f_small, dx, dy)
        print("Success! Got result:", len(grad_flat), "elements")
        
    except Exception as e:
        print("Error with 3x3:", e)
        import traceback
        traceback.print_exc()
        return
    
    # Now try 5x5
    print("\nTesting with 5x5 array...")
    try:
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f2d = X**2 + Y**2
        
        print("f2d shape:", f2d.shape)
        print("f2d content:")
        print(f2d)
        
        dx = 1.0
        dy = 1.0
        
        print("Calling gradient_2d_order2 with 5x5...")
        grad_flat = mygrad.gradient_2d_order2(f2d.tolist(), dx, dy)
        print("Success! Order 2 result:", len(grad_flat), "elements")
        
        print("Calling gradient_2d_order4 with 5x5...")
        grad_flat = mygrad.gradient_2d_order4(f2d.tolist(), dx, dy)
        print("Success! Order 4 result:", len(grad_flat), "elements")
        
        print("Calling gradient_2d_order6 with 5x5...")
        grad_flat = mygrad.gradient_2d_order6(f2d.tolist(), dx, dy)
        print("Success! Order 6 result:", len(grad_flat), "elements")
        
    except Exception as e:
        print("Error with 5x5:", e)
        import traceback
        traceback.print_exc()
        return
    
    # Now try 7x7
    print("\nTesting with 7x7 array...")
    try:
        x = np.linspace(-3, 3, 7)
        y = np.linspace(-3, 3, 7)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f2d = X**2 + Y**2
        
        print("f2d shape:", f2d.shape)
        dx = 1.0
        dy = 1.0
        
        print("Calling gradient_2d_order6 with 7x7...")
        grad_flat = mygrad.gradient_2d_order6(f2d.tolist(), dx, dy)
        print("Success! Order 6 result:", len(grad_flat), "elements")
        
        # If we get here, let's actually parse the results
        ny, nx = f2d.shape
        grad_x = np.array(grad_flat[:nx*ny]).reshape(ny, nx)
        grad_y = np.array(grad_flat[nx*ny:]).reshape(ny, nx)
        
        print("grad_x shape:", grad_x.shape)
        print("grad_y shape:", grad_y.shape)
        print("grad_x:")
        print(grad_x)
        print("grad_y:")
        print(grad_y)
        
    except Exception as e:
        print("Error with 7x7:", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_1d_gradient()
    test_2d_gradient_debug()