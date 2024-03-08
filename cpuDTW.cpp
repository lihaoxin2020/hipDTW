#include "utils.h"
#include <cmath>
#include <random>

#define REF_LENGTH 100000  // 100k
#define BATCH_SIZE 512
#define QUERY_LENGTH 2000

// Calculate mean and stdDev in batch, and normalize given data. 
void _normalizeData(float* data, const int size, float& mean, float& stdDev) {
    float sum = 0.0;
    float sumSq = 0.0;
    for (int i = 0; i < size; i++) {
        float value = data[i];
        sum += value;
        sumSq += value * value;
    }
    mean = sum / size;
    float variance = (sumSq / size) - (mean * mean);
    stdDev = sqrt(variance);

    // Normalize the data
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / stdDev;
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
    std::vector<float> ref = _generateRandomFloats(58, 120, REF_LENGTH, 42);
    std::vector<float> query = _generateRandomFloats(58, 120, QUERY_LENGTH * BATCH_SIZE, 42);

    float refMean, refStdDev;
    _normalizeData(ref.data(), REF_LENGTH, refMean, refStdDev);
    std::cout << "\nRef Mean: " << refMean << ", Standard Deviation: " << refStdDev << std::endl;

    std::string refFilename = "reference.txt";
    // Write to file
    writeDataToFile(refFilename, ref);

    std::string queryFilename = "query.txt";
    // Write to file
    writeDataToFile(queryFilename, query);

    float mean, stdDev;
    // _normalizeData(query, mean, stdDev);
    // std::cout << "Normalized data: ";
    // for (auto value : query) {
    //     std::cout << value << " ";
    // }
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        _normalizeData(query.data()+i*QUERY_LENGTH, QUERY_LENGTH, mean, stdDev);
    }
    std::cout << "Query Mean: " << mean << ", Standard Deviation: " << stdDev << std::endl;
    
    std::string queryNormFilename = "query-norm.txt";
    // Write to file
    writeDataToFile(queryNormFilename, query);

    // CPU sDTW
    std::vector<float> scores(BATCH_SIZE);
    // Organize queries in a batch of 512
    for (int i = 0; i < BATCH_SIZE; i++) {
        std::vector<float> singleQuery = std::vector<float>(query.begin() + (i*QUERY_LENGTH), query.begin() + (i+1)*QUERY_LENGTH);
        scores[i] = cpusDTW(singleQuery, ref);
        std::cout << "Batch " << i << "score: " << scores[i] << std::endl;
    }
    std::string cpuResultFilename = "cpusDTW.txt";
    // Write Results
    writeDataToFile(cpuResultFilename, scores);

    return 0;
}
