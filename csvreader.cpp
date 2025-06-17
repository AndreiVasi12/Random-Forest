#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


void readCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<int>& y)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        if (!row.empty()) {
            // Last value is the label (cast to int)
            y.push_back(static_cast<int>(row.back()));
            row.pop_back(); // remove label from features
            X.push_back(row);
        }
    }
}
