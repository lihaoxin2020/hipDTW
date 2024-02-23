#include <iostream>
#include <vector>
#include <fstream>

std::vector<float> readDataFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return std::vector<float>();
    }

    std::vector<float> data;
    float value;
    while (inFile >> value) {
        data.push_back(value);
    }

    inFile.close();

    // Print the data read from file
    // std::cout << "Data read from " << filename << ":" << std::endl;
    // for (auto val : data) {
    //     std::cout << val << std::endl;
    // }
    return data;
}

// Write to file
void writeDataToFile(const std::string& filename, std::vector<float> data) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (auto value : data) {
        outFile << value << std::endl;
    }
    outFile.close();
}
