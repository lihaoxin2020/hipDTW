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

#define BLOCK_SIZE2 1000
#define NUM_BATCHES 512

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


// might need to add small epsilon to avoid division by zero
__global__ void computeMeansAndStdDevs2(float* input, float* means, float* stdDevs, int valuesPerQuery) {
    __shared__ float sdata[BLOCK_SIZE2];
    __shared__ float mean;
    __shared__ float stdDev;


    // int tid = threadIdx.x;
    // int i = blockIdx.x * blockDim.x + threadIdx.x;

    // if (i < valuesPerQuery) {
    //     sdata[tid] = input[i];
    // } else {
    //     sdata[tid] = 0;
    // }

    int tid = threadIdx.x;
    // if (tid >= 1000) {
    //     return;
    // }
    int i = threadIdx.x + blockIdx.x * 2 * blockDim.x;
    
    // if (i < )
    // sdata[tid] = input[i];
    // else
    //     sdata[tid] = 0.0;

    // if (i + blockDim.x < len) {
    sdata[tid] = input[i] + input[i + blockDim.x];
    // }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // if (tid == 0) {
    //     means[blockIdx.x] = sdata[0] / valuesPerQuery;
    // }

    // if (tid == 0) {
    //     printf("Sum: %f\n", sdata[0]);
    //     printf("Alt Sum: %f\n", sdata[0] + sdata[124] + sdata[30] + sdata[14] + sdata[6] + sdata[2]);
    // }
    
    if (tid == 0) {
        sdata[0] += (sdata[124] + sdata[30] + sdata[14] + sdata[6] + sdata[2]);
        mean = sdata[0] / valuesPerQuery;
    }

    __syncthreads();

    // if (i < valuesPerQuery) {
    sdata[tid] = (input[i] - mean) * (input[i] - mean);
    sdata[tid] += (input[i + blockDim.x] - mean) * (input[i + blockDim.x] - mean);
    // } else {
    //     sdata[tid] = 0;
    // }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] += (sdata[124] + sdata[30] + sdata[14] + sdata[6] + sdata[2]);
        stdDev = sqrt(sdata[0] / valuesPerQuery);
    }

    __syncthreads();

    // if (i < valuesPerQuery) {
    input[i] = (input[i] - mean) / stdDev;
    input[i + blockDim.x] = (input[i + blockDim.x] - mean) / stdDev;
    // }

    if (tid == 0) {
        means[blockIdx.x] = mean;
        stdDevs[blockIdx.x] = stdDev;
        // means
        // std::cout << "Block: " << blockIdx.x << ", Mean: " << mean << ", Standard Deviation: " << stdDev << std::endl;
        
        // std::cout << "Block: " << blockIdx.x << ", Mean: " << means[blockIdx.x] << ", Standard Deviation: " << stdDevs[blockIdx.x] << std::endl;
        int blockId = blockIdx.x;
        printf("Block: %d, Mean (mean): %f, Standard Deviation (std): %f\n", blockId, mean, stdDev);
        // printf("Block: %d, Mean: %f, Standard Deviation: %f\n", blockId, means[blockIdx.x], stdDevs[blockIdx.x]);
    }

    // printf("index 1: %d, index 2: %d\n", i, i + blockDim.x);


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

    // std::cout << "Query length: " << queryLength << std::endl;
    printf("Query length: %zu\n", queryLength);
    // hipLaunchKernelGGL(
    //     computeMeansAndStdDevs, 
    //     dim3 (1), // Only 1 block needed
    //     dim3 (BLOCK_SIZE), 
    //     0, 0, 
    //     deviceQuery, means, stdDevs, queryLength
    // );

    hipLaunchKernelGGL(
        computeMeansAndStdDevs2, 
        dim3 (NUM_BATCHES,1,1), // Only 1 block needed
        dim3 (BLOCK_SIZE2,1,1), 
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
