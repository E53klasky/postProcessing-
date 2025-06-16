#pragma once
#include <vector>
#include <cstddef>

std::vector<double> gradient_1d_order2(const std::vector<double>& f , double dx);
std::vector<double> gradient_1d_order4(const std::vector<double>& f , double dx);
std::vector<double> gradient_1d_order6(const std::vector<double>& f , double dx);
