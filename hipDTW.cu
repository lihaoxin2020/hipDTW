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
#define INF 1e20
#define k 4
#define p 500

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

__global__ void dtw(float *subjects, float *scores, int m, float *cQ) {
    float S[k]; // Registers storing subject data
    float M[k]; // Registers storing DP matrix cells
    float M_left, M_diag; // Registers for data from left neighbor
    float Reg_Q0, Reg_Q1; // Registers storing query data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k; i++) {
        S[i] = subjects[idx * k + i];
    }

    if (idx % p == 0) {
        M[0] = (subjects[idx] - cQ[0]) * (subjects[idx] - cQ[0]);
    } else {
        M[0] = 
    }
    // init rest of first row. 0 init as cpu implementation
    if (idx_x > 0) {
        s[idx_y][0][idx_x] = 0;
    }
    // load_subject(S, subjects, threadIdx.x + blockIdx.x * k, k, p);

    
    // Load one query value per thread
    Reg_Q1 = cQ[threadIdx.x % p];
    if (threadIdx.x % p == 0) Reg_Q0 = Reg_Q1;
    else Reg_Q0 = INFTY;
    
    for (int i = 1; i < k+p; i++) { // Wavefront loop
        // Compute DP cells per thread using registers only
        // ...

        // Copy rightmost DP cell to neighboring thread
        M_left = __shfl_up_sync(0xffffffff, M[k-1], 1);
        if (threadIdx.x % p == 0) M_left = INFTY;
        
        // Load new query data to register every p iterations
        if (i % p == 0) Reg_Q1 = cQ[i/p + threadIdx.x % p];
        
        // Shuffle query registers
        Reg_Q0 = __shfl_up_sync(0xffffffff, Reg_Q0, 1);
        if (threadIdx.x % p == 0) Reg_Q0 = Reg_Q1;
        Reg_Q1 = __shfl_down_sync(0xffffffff, Reg_Q1, 1);
        
        // ...
    }
    
    output_DTW_score(M[k-1], scores, threadIdx.x + blockIdx.x * k, k, p);
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
    std::string queryFilename = argv[1];
    std::string refFilename = argv[2];
    std::vector<float> data = readDataFromFile(queryFilename);
    std::vector<float> ref = readDataFromFile(refFilename);
    
    size_t queryLength = data.size() / BATCH_SIZE;
    size_t refLength = ref.size();

    float *query = data.data();
    float *hostRef = ref.data();
    float *deviceQuery, *deviceRef, *deviceScores, *scoreMetrics;
    size_t sizeData = data.size() * sizeof(float);
    size_t sizeRef = ref.size() * sizeof(float);

    wbCheck(hipMalloc((void **) &deviceQuery, sizeData));
    wbCheck(hipMalloc((void **) &deviceRef, sizeRef));
    wbCheck(hipMemcpy(deviceQuery, query, sizeData, hipMemcpyHostToDevice));
    wbCheck(hipMemcpy(deviceRef, hostRef, sizeRef, hipMemcpyHostToDevice));

    float *means, *stdDevs;
    size_t statsSize = BATCH_SIZE * sizeof(float);
    wbCheck(hipMalloc((void **) &means, statsSize));
    wbCheck(hipMalloc((void **) &stdDevs, statsSize));
    wbCheck(hipMalloc((void **) &deviceScores, statsSize));
    wbCheck(hipMalloc((void **) &scoreMetrics, statsSize * queryLength * refLength));

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
    
    std::cout << "Normalized. Starting sDTW..." << std::endl;
    dim3 threadsPerBlock(16, 16); // Adjust as needed
    dim3 numBlocks((refLength + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t sharedMemSize = BATCH_SIZE * queryLength * refLength * sizeof(float);

    hipLaunchKernelGGL(
        sDTW, 
        numBlocks, 
        threadsPerBlock, 
        0, 0, 
        deviceQuery, deviceRef, queryLength, refLength, scoreMetrics, deviceScores
    );

    // Copy means stdDevs to host
    float *hostMeans = new float[BATCH_SIZE];
    float *hostStdDevs = new float[BATCH_SIZE];
    float *normalizedQueries = new float[data.size()];
    float *hostScores = new float[BATCH_SIZE];
    wbCheck(hipMemcpy(hostMeans, means, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostStdDevs, stdDevs, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(normalizedQueries, deviceQuery, sizeData, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostScores, deviceScores, statsSize, hipMemcpyDeviceToHost));

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "Mean: " << hostMeans[i] << ", Standard Deviation: " << hostStdDevs[i] << std::endl;
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "sDTW scores: " << hostScores[i] << std::endl;
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
    delete[] hostScores;
    wbCheck(hipFree(deviceQuery));
    wbCheck(hipFree(means));
    wbCheck(hipFree(stdDevs));
    wbCheck(hipFree(deviceRef));
    wbCheck(hipFree(scoreMetrics));
    wbCheck(hipFree(deviceScores));

    return 0;
}
