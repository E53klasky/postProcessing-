#include "gradient.h"
#include <stdexcept>
#include <vector>

// note you need to make it return a 2d vector  all grarbarge redo -----------------------------------------------------------


// 2nd order accurate gradient in 2D
// Returns a flattened 1D vector of size 2 * nx * ny
// First nx*ny entries: ∂f/∂x, second nx*ny entries: ∂f/∂y
std::vector<double> gradient_2d_order2(const std::vector<std::vector<double>>& f , double dx , double dy) {
    size_t ny = f.size();
    if (ny == 0) throw std::invalid_argument("Input 2D vector f is empty.");
    size_t nx = f[0].size();
    for (const auto& row : f)
        if (row.size() != nx)
            throw std::invalid_argument("All rows in f must have the same number of elements.");

    std::vector<double> grad(2 * nx * ny , 0.0);
    if (nx < 3 || ny < 3) return grad; // Require at least 3 points in both directions

    // ∂f/∂x (row-by-row)
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            size_t idx = j * nx + i;
            if (i == 0) {
                grad[idx] = (-3 * f[j][i] + 4 * f[j][i + 1] - f[j][i + 2]) / (2 * dx);
            }
            else if (i == nx - 1) {
                grad[idx] = (3 * f[j][i] - 4 * f[j][i - 1] + f[j][i - 2]) / (2 * dx);
            }
            else {
                grad[idx] = (f[j][i + 1] - f[j][i - 1]) / (2 * dx);
            }
        }
    }

    // ∂f/∂y (column-by-column), stored in second half
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            size_t idx = j * nx + i;
            size_t outIdx = nx * ny + idx;
            if (j == 0) {
                grad[outIdx] = (-3 * f[j][i] + 4 * f[j + 1][i] - f[j + 2][i]) / (2 * dy);
            }
            else if (j == ny - 1) {
                grad[outIdx] = (3 * f[j][i] - 4 * f[j - 1][i] + f[j - 2][i]) / (2 * dy);
            }
            else {
                grad[outIdx] = (f[j + 1][i] - f[j - 1][i]) / (2 * dy);
            }
        }
    }

    return grad;
}

// 4th order accurate gradient (O(h^4)) in 2D.
// Computes gradients in x and y directions for a flattened 2D array.
// Fallback to 2nd order when there are not enough points in a given direction.
std::vector<double> gradient_2d_order4(const std::vector<std::vector<double>>& f , double dx , double dy) {
    size_t ny = f.size();
    if (ny == 0) throw std::invalid_argument("Input 2D vector f is empty.");
    size_t nx = f[0].size();
    for (const auto& row : f)
        if (row.size() != nx)
            throw std::invalid_argument("All rows in f must have the same number of elements.");


    std::vector<double> grad(2 * nx * ny , 0.0); // First nx*ny entries: ∂f/∂x, next nx*ny: ∂f/∂y

    // ∂f/∂x (row-wise)
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            size_t idx = j * nx + i;

            if (nx < 5) {
                // Fallback to 2nd order
                if (i == 0)
                    grad[idx] = (-3 * f[j][i] + 4 * f[j][i + 1] - f[j][i + 2]) / (2 * dx);
                else if (i == nx - 1)
                    grad[idx] = (3 * f[j][i] - 4 * f[j][i - 1] + f[j][i - 2]) / (2 * dx);
                else
                    grad[idx] = (f[j][i + 1] - f[j][i - 1]) / (2 * dx);
            }
            else {
                if (i == 0)
                    grad[idx] = (-25 * f[j][i] + 48 * f[j][i + 1] - 36 * f[j][i + 2]
                        + 16 * f[j][i + 3] - 3 * f[j][i + 4]) / (12 * dx);
                else if (i == 1)
                    grad[idx] = (-3 * f[j][0] - 10 * f[j][1] + 18 * f[j][2]
                        - 6 * f[j][3] + f[j][4]) / (12 * dx);
                else if (i >= 2 && i < nx - 2)
                    grad[idx] = (-f[j][i + 2] + 8 * f[j][i + 1]
                        - 8 * f[j][i - 1] + f[j][i - 2]) / (12 * dx);
                else if (i == nx - 2)
                    grad[idx] = (-f[j][i - 2] + 6 * f[j][i - 1]
                        - 18 * f[j][i] + 10 * f[j][i + 1]
                        + 3 * f[j][i + 2]) / (12 * dx);
                else if (i == nx - 1)
                    grad[idx] = (3 * f[j][i - 4] - 16 * f[j][i - 3]
                        + 36 * f[j][i - 2] - 48 * f[j][i - 1]
                        + 25 * f[j][i]) / (12 * dx);
            }
        }
    }

    // ∂f/∂y (column-wise)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            size_t idx = j * nx + i;
            size_t outIdx = nx * ny + idx; // gradient_y is stored after gradient_x

            if (ny < 5) {
                // Fallback to 2nd order
                if (j == 0)
                    grad[outIdx] = (-3 * f[0][i] + 4 * f[1][i] - f[2][i]) / (2 * dy);
                else if (j == ny - 1)
                    grad[outIdx] = (3 * f[ny - 1][i] - 4 * f[ny - 2][i] + f[ny - 3][i]) / (2 * dy);
                else
                    grad[outIdx] = (f[j + 1][i] - f[j - 1][i]) / (2 * dy);
            }
            else {
                if (j == 0)
                    grad[outIdx] = (-25 * f[0][i] + 48 * f[1][i] - 36 * f[2][i]
                        + 16 * f[3][i] - 3 * f[4][i]) / (12 * dy);
                else if (j == 1)
                    grad[outIdx] = (-3 * f[0][i] - 10 * f[1][i] + 18 * f[2][i]
                        - 6 * f[3][i] + f[4][i]) / (12 * dy);
                else if (j >= 2 && j < ny - 2)
                    grad[outIdx] = (-f[j + 2][i] + 8 * f[j + 1][i]
                        - 8 * f[j - 1][i] + f[j - 2][i]) / (12 * dy);
                else if (j == ny - 2)
                    grad[outIdx] = (-f[j - 2][i] + 6 * f[j - 1][i]
                        - 18 * f[j][i] + 10 * f[j + 1][i]
                        + 3 * f[j + 2][i]) / (12 * dy);
                else if (j == ny - 1)
                    grad[outIdx] = (3 * f[j - 4][i] - 16 * f[j - 3][i]
                        + 36 * f[j - 2][i] - 48 * f[j - 1][i]
                        + 25 * f[j][i]) / (12 * dy);
            }
        }
    }

    return grad;
}


// 6th order accurate gradient (O(h^6)) in 2D.
// Computes gradients in x and y directions for a flattened 2D array.
// Fallback to 4th order when there are not enough points in a given direction.
std::vector<double> gradient_2d_order6(const std::vector<std::vector<double>>& f , double dx , double dy) {
    size_t ny = f.size();
    if (ny == 0) throw std::invalid_argument("Input 2D vector f is empty.");
    size_t nx = f[0].size();
    for (const auto& row : f)
        if (row.size() != nx)
            throw std::invalid_argument("All rows in f must have the same number of elements.");

    std::vector<double> grad(2 * nx * ny , 0.0);

    // ∂f/∂x
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            size_t idx = j * nx + i;

            if (nx < 7) {
                // fallback to 4th or 2nd order
                if (nx < 5) {
                    // 2nd order
                    if (i == 0)
                        grad[idx] = (-3 * f[j][i] + 4 * f[j][i + 1] - f[j][i + 2]) / (2 * dx);
                    else if (i == nx - 1)
                        grad[idx] = (3 * f[j][i] - 4 * f[j][i - 1] + f[j][i - 2]) / (2 * dx);
                    else
                        grad[idx] = (f[j][i + 1] - f[j][i - 1]) / (2 * dx);
                }
                else {
                    // 4th order
                    if (i == 0)
                        grad[idx] = (-25 * f[j][0] + 48 * f[j][1] - 36 * f[j][2] + 16 * f[j][3] - 3 * f[j][4]) / (12 * dx);
                    else if (i == 1)
                        grad[idx] = (-3 * f[j][0] - 10 * f[j][1] + 18 * f[j][2] - 6 * f[j][3] + f[j][4]) / (12 * dx);
                    else if (i >= 2 && i < nx - 2)
                        grad[idx] = (-f[j][i + 2] + 8 * f[j][i + 1] - 8 * f[j][i - 1] + f[j][i - 2]) / (12 * dx);
                    else if (i == nx - 2)
                        grad[idx] = (-f[j][i - 2] + 6 * f[j][i - 1] - 18 * f[j][i] + 10 * f[j][i + 1] + 3 * f[j][i + 2]) / (12 * dx);
                    else
                        grad[idx] = (3 * f[j][i - 4] - 16 * f[j][i - 3] + 36 * f[j][i - 2] - 48 * f[j][i - 1] + 25 * f[j][i]) / (12 * dx);
                }
            }
            else {
                // 6th order - FIXED boundary conditions
                if (i < 3) {
                    // Forward stencil for first 3 points
                    grad[idx] = (-147 * f[j][i] + 360 * f[j][i + 1] - 450 * f[j][i + 2]
                        + 400 * f[j][i + 3] - 225 * f[j][i + 4] + 72 * f[j][i + 5] - 10 * f[j][i + 6]) / (60 * dx);
                }
                else if (i >= nx - 3) {
                    // Backward stencil for last 3 points
                    grad[idx] = (10 * f[j][i - 6] - 72 * f[j][i - 5] + 225 * f[j][i - 4]
                        - 400 * f[j][i - 3] + 450 * f[j][i - 2] - 360 * f[j][i - 1] + 147 * f[j][i]) / (60 * dx);
                }
                else {
                    // Central stencil for interior points (i = 3, 4, ..., nx-4)
                    grad[idx] = (f[j][i - 3] - 9 * f[j][i - 2] + 45 * f[j][i - 1]
                        - 45 * f[j][i + 1] + 9 * f[j][i + 2] - f[j][i + 3]) / (60 * dx);
                }
            }
        }
    }

    // ∂f/∂y
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            size_t idx = j * nx + i;
            size_t outIdx = nx * ny + idx;

            if (ny < 7) {
                // fallback to 4th or 2nd order
                if (ny < 5) {
                    // 2nd order
                    if (j == 0)
                        grad[outIdx] = (-3 * f[j][i] + 4 * f[j + 1][i] - f[j + 2][i]) / (2 * dy);
                    else if (j == ny - 1)
                        grad[outIdx] = (3 * f[j][i] - 4 * f[j - 1][i] + f[j - 2][i]) / (2 * dy);
                    else
                        grad[outIdx] = (f[j + 1][i] - f[j - 1][i]) / (2 * dy);
                }
                else {
                    // 4th order
                    if (j == 0)
                        grad[outIdx] = (-25 * f[0][i] + 48 * f[1][i] - 36 * f[2][i] + 16 * f[3][i] - 3 * f[4][i]) / (12 * dy);
                    else if (j == 1)
                        grad[outIdx] = (-3 * f[0][i] - 10 * f[1][i] + 18 * f[2][i] - 6 * f[3][i] + f[4][i]) / (12 * dy);
                    else if (j >= 2 && j < ny - 2)
                        grad[outIdx] = (-f[j + 2][i] + 8 * f[j + 1][i] - 8 * f[j - 1][i] + f[j - 2][i]) / (12 * dy);
                    else if (j == ny - 2)
                        grad[outIdx] = (-f[j - 2][i] + 6 * f[j - 1][i] - 18 * f[j][i] + 10 * f[j + 1][i] + 3 * f[j + 2][i]) / (12 * dy);
                    else
                        grad[outIdx] = (3 * f[j - 4][i] - 16 * f[j - 3][i] + 36 * f[j - 2][i] - 48 * f[j - 1][i] + 25 * f[j][i]) / (12 * dy);
                }
            }
            else {
                // 6th order - FIXED boundary conditions
                if (j < 3) {
                    // Forward stencil for first 3 points
                    grad[outIdx] = (-147 * f[j][i] + 360 * f[j + 1][i] - 450 * f[j + 2][i]
                        + 400 * f[j + 3][i] - 225 * f[j + 4][i] + 72 * f[j + 5][i] - 10 * f[j + 6][i]) / (60 * dy);
                }
                else if (j >= ny - 3) {
                    // Backward stencil for last 3 points
                    grad[outIdx] = (10 * f[j - 6][i] - 72 * f[j - 5][i] + 225 * f[j - 4][i]
                        - 400 * f[j - 3][i] + 450 * f[j - 2][i] - 360 * f[j - 1][i] + 147 * f[j][i]) / (60 * dy);
                }
                else {
                    // Central stencil for interior points (j = 3, 4, ..., ny-4)
                    grad[outIdx] = (f[j - 3][i] - 9 * f[j - 2][i] + 45 * f[j - 1][i]
                        - 45 * f[j + 1][i] + 9 * f[j + 2][i] - f[j + 3][i]) / (60 * dy);
                }
            }
        }
    }

    return grad;
}