#include <hip/hip_runtime.h>
#include <math.h>
#include <wb.h>


#include "utils.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 512 // Adjust based on your GPU's architecture
#define BATCH_SIZE 512 // Normalizer batch size

// Naive version
// Input 512 queries of valuesPerQuery each. 
// Output 512 means and std devs, and normalized queries in place.
__global__ void computeMeansAndStdDevs(float* input, float* means, float* stdDevs, int valuesPerQuery) {
    int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int values = valuesPerQuery;

    if (queryIdx < BLOCK_SIZE) {
        float sum = 0;
        for (int i = 0; i < values; ++i) {
            sum += input[queryIdx * values + i];
        }
        float mean = sum / values;
        means[queryIdx] = mean;

        float sumSqDiff = 0;
        for (int i = 0; i < values; ++i) {
            float diff = input[queryIdx * values + i] - mean;
            sumSqDiff += diff * diff;
        }
        float variance = sumSqDiff / values;
        stdDevs[queryIdx] = sqrt(variance);

        for (int i = 0; i < values; ++i) {
            input[queryIdx * values + i] -= means[queryIdx];
            input[queryIdx * values + i] /= stdDevs[queryIdx];
        }
    }
}

int main(int argc, char const *argv[])
{
    std::string filename = argv[1];
    std::vector<float> data = readDataFromFile(filename);
    size_t queryLength = data.size() / BATCH_SIZE;

    float *query = data.data();
    float *deviceQuery;
    size_t sizeData = data.size() * sizeof(float);

    wbCheck(hipMalloc((void **) &deviceQuery, sizeData));
    wbCheck(hipMemcpy(deviceQuery, query, sizeData, hipMemcpyHostToDevice));

    float *means, *stdDevs;
    size_t statsSize = BATCH_SIZE * sizeof(float);
    wbCheck(hipMalloc((void **) &means, statsSize));
    wbCheck(hipMalloc((void **) &stdDevs, statsSize));

    std::cout << "Starting kernel..." << std::endl;
    hipLaunchKernelGGL(
        computeMeansAndStdDevs, 
        dim3 (1), // Only 1 block needed
        dim3 (BLOCK_SIZE), 
        0, 0, 
        deviceQuery, means, stdDevs, queryLength
    );

    // Copy means stdDevs to host
    float *hostMeans = new float[BATCH_SIZE];
    float *hostStdDevs = new float[BATCH_SIZE];
    float *normalizedQueries = new float[data.size()];
    wbCheck(hipMemcpy(hostMeans, means, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostStdDevs, stdDevs, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(normalizedQueries, deviceQuery, sizeData, hipMemcpyDeviceToHost));

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "Mean: " << hostMeans[i] << ", Standard Deviation: " << hostStdDevs[i] << std::endl;
    }

    // float cpuSum = 0;
    // for (auto value:data){
    //     cpuSum += value;
    // }
    // float cpuMean = cpuSum / queryLength;
    // std::cout << "CPU Results: " << std::endl;
    // std::cout << "Total sum: " << cpuSum;
    // std::cout << "Mean: " << cpuMean << std::endl;

    delete[] hostMeans;
    delete[] hostStdDevs;
    delete[] normalizedQueries;
    wbCheck(hipFree(deviceQuery));
    wbCheck(hipFree(means));
    wbCheck(hipFree(stdDevs));

    return 0;
}
