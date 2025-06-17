#pragma once
#include <vector>
#include <string>

// Declare the CSV reading function
void readCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<int>& y);
