#pragma once
#include <vector>
#include <cstddef>

std::vector<double> gradient_1d_order2(const std::vector<double>& f , double dx);
std::vector<double> gradient_1d_order4(const std::vector<double>& f , double dx);
std::vector<double> gradient_1d_order6(const std::vector<double>& f , double dx);

std::vector<double> gradient_2d_order2(const std::vector<std::vector<double>>& f , double dx , double dy);
std::vector<double> gradient_2d_order4(const std::vector<std::vector<double>>& f , double dx , double dy);
std::vector<double> gradient_2d_order6(const std::vector<std::vector<double>>& f , double dx , double dy);

// std::vector<double> gradient_3d_order2(const std::vector<double>& f , double dx , double dy , double dz);
// std::vector<double> gradient_3d_order4(const std::vector<double>& f , double dx , double dy , double dz);
// std::vector<double> gradient_3d_order6(const std::vector<double>& f , double dx , double dy , double dz);

// std::vector<double> gradient_4d_order2(const std::vector<double>& f , double dx , double dy , double dz);
// std::vector<double> gradient_4d_order4(const std::vector<double>& f , double dx , double dy , double dz);
// std::vector<double> gradient_4d_order6(const std::vector<double>& f , double dx , double dy , double dz);
