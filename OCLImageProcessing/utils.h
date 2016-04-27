#pragma once

#include <string>

std::string FileToString(const std::string fileName);

// create a normalized Gaussian mask
void createGaussianFilter(float *kernel, int size, float sd);