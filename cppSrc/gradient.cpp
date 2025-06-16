#include "gradient.h"
#include <stdexcept>
#include <vector>

// 2nd order accurate gradient (O(h^2))
std::vector<double> gradient_1d_order2(const std::vector<double>& f , double dx) {
    size_t n = f.size();
    std::vector<double> grad(n , 0.0);
    if (n < 2) return grad;

    // edges: 2nd order one-sided difference
    grad[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx);
    grad[n - 1] = (3 * f[n - 1] - 4 * f[n - 2] + f[n - 3]) / (2 * dx);

    // interior: central difference
    for (size_t i = 1; i < n - 1; ++i) {
        grad[i] = (f[i + 1] - f[i - 1]) / (2 * dx);
    }

    return grad;
}

// 4th order accurate gradient (O(h^4))
std::vector<double> gradient_1d_order4(const std::vector<double>& f , double dx) {
    size_t n = f.size();
    std::vector<double> grad(n , 0.0);
    if (n < 5) // Need at least 5 points for 4th order edges
        return gradient_1d_order2(f , dx); // fallback

    // edges: 4th order one-sided differences
    grad[0] = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * dx);
    grad[1] = (-3 * f[0] - 10 * f[1] + 18 * f[2] - 6 * f[3] + f[4]) / (12 * dx);

    grad[n - 2] = (-f[n - 5] + 6 * f[n - 4] - 18 * f[n - 3] + 10 * f[n - 2] + 3 * f[n - 1]) / (12 * dx);
    grad[n - 1] = (3 * f[n - 5] - 16 * f[n - 4] + 36 * f[n - 3] - 48 * f[n - 2] + 25 * f[n - 1]) / (12 * dx);

    // interior: 4th order central difference
    for (size_t i = 2; i < n - 2; ++i) {
        grad[i] = (-f[i + 2] + 8 * f[i + 1] - 8 * f[i - 1] + f[i - 2]) / (12 * dx);
    }

    return grad;
}

// 6th order accurate gradient (O(h^6))
std::vector<double> gradient_1d_order6(const std::vector<double>& f , double dx) {
    size_t n = f.size();
    std::vector<double> grad(n , 0.0);
    if (n < 7) // Need at least 7 points for 6th order edges
        return gradient_1d_order4(f , dx); // fallback

    // edges: 6th order one-sided differences
    grad[0] = (-147 * f[0] + 360 * f[1] - 450 * f[2] + 400 * f[3] - 225 * f[4] + 72 * f[5] - 10 * f[6]) / (60 * dx);
    grad[1] = (-10 * f[0] - 77 * f[1] + 150 * f[2] - 100 * f[3] + 50 * f[4] - 15 * f[5] + 2 * f[6]) / (60 * dx);
    grad[2] = (2 * f[0] - 24 * f[1] - 35 * f[2] + 80 * f[3] - 30 * f[4] + 8 * f[5] - f[6]) / (60 * dx);

    grad[n - 3] = (-f[n - 7] + 8 * f[n - 6] - 30 * f[n - 5] + 80 * f[n - 4] - 35 * f[n - 3] - 24 * f[n - 2] + 2 * f[n - 1]) / (60 * dx);
    grad[n - 2] = (2 * f[n - 7] - 15 * f[n - 6] + 50 * f[n - 5] - 100 * f[n - 4] + 150 * f[n - 3] - 77 * f[n - 2] - 10 * f[n - 1]) / (60 * dx);
    grad[n - 1] = (10 * f[n - 7] - 72 * f[n - 6] + 225 * f[n - 5] - 400 * f[n - 4] + 450 * f[n - 3] - 360 * f[n - 2] + 147 * f[n - 1]) / (60 * dx);

    // interior: 6th order central difference
    for (size_t i = 3; i < n - 3; ++i) {
        grad[i] = (f[i - 3] - 9 * f[i - 2] + 45 * f[i - 1] - 45 * f[i + 1] + 9 * f[i + 2] - f[i + 3]) / (60 * dx);
    }

    return grad;
}

