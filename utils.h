#include <iostream>
#include <vector>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

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

// 
float cpusDTW(const std::vector<float>& Q, const std::vector<float>& R) {
    size_t N = Q.size();
    size_t M = R.size();
    
    std::vector<std::vector<float>> S(N, std::vector<float>(M, 0));

    // Initialize the first column of S
    S[0][0] = (Q[0] - R[0]) * (Q[0] - R[0]);
    // for (size_t i = 1; i < M; ++i) {
    //     S[0][i] = (Q[0] - R[i]) * (Q[0] - R[i]);
    // }
    for (size_t i = 1; i < N; ++i) {
        S[i][0] = S[i - 1][0] + (Q[i] - R[0]) * (Q[i] - R[0]);
    }

    // Populate the rest of S
    for (size_t i = 1; i < N; ++i) {
        for (size_t j = 1; j < M; ++j) {
            float cost = (Q[i] - R[j]) * (Q[i] - R[j]);
            S[i][j] = cost + std::min({S[i - 1][j - 1], S[i][j - 1], S[i - 1][j]});
        }
    }

    // Find the minimum value in the last row of S
    std::vector<float> lastRow(S[N - 1].begin(), S[N - 1].end());
    return *std::min_element(lastRow.begin(), lastRow.end());
}
