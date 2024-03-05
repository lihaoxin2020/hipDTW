#include "utils.h"
#include <cmath>
#include <random>

#define REF_LENGTH 100000  // 100k
#define QUERY_LENGTH 2000 // 512 * 2k = 1,024k

void _normalizeData(std::vector<float>& data, float& mean, float& stdDev) {
    float sum = 0.0;
    float sumSq = 0.0;
    for (auto value : data) {
        sum += value;
        sumSq += value * value;
    }
    mean = sum / data.size();
    float variance = (sumSq / data.size()) - (mean * mean);
    stdDev = sqrt(variance);

    // Normalize the data
    for (auto& value : data) {
        value = (value - mean) / stdDev;
    }
}

std::vector<float> _generateRandomFloats(float min, float max, int size, unsigned seed) {
    std::mt19937 eng(seed); // Use a fixed seed for reproducibility
    std::uniform_real_distribution<> distr(min, max); // Define the range

    std::vector<float> randomFloats;
    for(int n = 0; n < size; ++n) {
        randomFloats.push_back(distr(eng)); // Generate a random float and add it to the vector
    }

    return randomFloats;
}

int main() {
    // Example usage
    std::vector<float> ref = _generateRandomFloats(-1, 1, REF_LENGTH, 42);
    std::vector<float> query = _generateRandomFloats(58, 120, QUERY_LENGTH, 42);

    // std::string refFilename = "reference.bin";
    // // Write to file
    // writeDataToFile(refFilename, ref);

    std::string queryFilename = "query2.bin";
    // Write to file
    writeDataToFile(queryFilename, query);

    float mean, stdDev;
    _normalizeData(query, mean, stdDev);
    // std::cout << "Normalized data: ";
    // for (auto value : query) {
    //     std::cout << value << " ";
    // }
    std::cout << "\nMean: " << mean << ", Standard Deviation: " << stdDev << std::endl;
    
    std::string queryNormFilename = "query2-norm.bin";
    // Write to file
    writeDataToFile(queryNormFilename, query);

    return 0;
}
