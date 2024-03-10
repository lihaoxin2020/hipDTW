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
#define REF_LENGTH 100000

#define BLOCK_SIZE2 1000
#define NUM_BATCHES 512

#define INF 1e20
#define k 1563
#define p 64

__constant__ float cQ[2000 * BATCH_SIZE];

// Input 512 queries of valuesPerQuery each. 
// Output 512 means and std devs, and normalized queries in place.
// might need to add small epsilon to avoid division by zero
__global__ void computeMeansAndStdDevs2(float* input, float* means, float* stdDevs, int valuesPerQuery) {
    __shared__ float sdata[BLOCK_SIZE2];
    __shared__ float mean;
    __shared__ float stdDev;

    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * 2 * blockDim.x;

    sdata[tid] = input[i] + input[i + blockDim.x];

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sdata[0] += (sdata[124] + sdata[30] + sdata[14] + sdata[6] + sdata[2]);
        mean = sdata[0] / valuesPerQuery;
    }

    __syncthreads();

    sdata[tid] = (input[i] - mean) * (input[i] - mean);
    sdata[tid] += (input[i + blockDim.x] - mean) * (input[i + blockDim.x] - mean);

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

    input[i] = (input[i] - mean) / stdDev;
    input[i + blockDim.x] = (input[i + blockDim.x] - mean) / stdDev;

    if (tid == 0) {
        means[blockIdx.x] = mean;
        stdDevs[blockIdx.x] = stdDev;
        // means
        int blockId = blockIdx.x;
    }
}

__global__ void dtw(float *subjects, float *scores, float *min_scores, int m) {
    float S[k]; // Registers storing subject data
    float M[k]; // Registers storing DP matrix cells
    float M_left, M_diag; // Registers for data from left neighbor
    float Reg_Q0, Reg_Q1; // Registers storing query data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load subjects
    for (int i = 0; i < k; i++) {
        S[i] = subjects[threadIdx.x * k + i];
    }
    // init DP matrix
    for (int i = 0; i < k; i++) {
        if (threadIdx.x % p == 0) M[i] = INF;
        else M[i] = 0;
    }
    M_diag = 0; M_left = INF;
    // Load one query value per thread
    Reg_Q1 = cQ[blockIdx.x * m + threadIdx.x % p];
    
    if (threadIdx.x % p == 0) {
        Reg_Q0 = Reg_Q1;
    }
    else Reg_Q0 = INF;
    Reg_Q1 = __shfl_down(Reg_Q1, 1);
    
    for (int i = 1; i <= m+p; i++) { // Wavefront loop
        // Compute DP cells per thread using registers only
        float upper;
        for (int j = 0; j < k; j++) {
            float temp;
            if (i - threadIdx.x == 1) temp = (S[j] - Reg_Q0) * (S[j] - Reg_Q0);
            else temp = fmin(M_left, fmin(M_diag, M[j])) + (S[j] - Reg_Q0) * (S[j] - Reg_Q0);
            M_diag = M[j];
            upper = M[j];
            M_left = temp;
            M[j] = temp;
        }
        M_diag = M_left;

        // Copy rightmost DP cell to neighboring thread
        __syncthreads();
        M_left = __shfl_up(M[k-1], 1);
        M_diag = __shfl_up(upper, 1);
        if (threadIdx.x % p == 0) {
            M_left = INF;
            M_diag = INF;
        }

        // Load new query data to register every p iterations
        if (i % p == 0 && i + threadIdx.x < m) Reg_Q1 = cQ[m * blockIdx.x + i + threadIdx.x % p];
        
        // Shuffle query registers
        __syncthreads();
        Reg_Q0 = __shfl_up(Reg_Q0, 1);
        __syncthreads();
        if (threadIdx.x % p == 0) 
            Reg_Q0 = Reg_Q1;
        
        Reg_Q1 = __shfl_down(Reg_Q1, 1);
        __syncthreads();
     
        if (i - threadIdx.x == m) {
            for (int j = 0; j < k && threadIdx.x * k + j < REF_LENGTH; j++) {
                scores[REF_LENGTH * blockIdx.x + threadIdx.x * k + j] = M[j];
            }
        }
    }
    
    if (threadIdx.x == 0) {
        float min_score = INF;
        for (int i = 0; i < 100000; i++) {
            min_score = fmin(scores[REF_LENGTH * blockIdx.x + i], min_score);
        }
        min_scores[blockIdx.x] = min_score;
    }
}


int main(int argc, char const *argv[])
{
    // Reading data
    std::string queryFilename = argv[1];
    std::string refFilename = argv[2];
    std::vector<float> data = readDataFromFile(queryFilename);
    std::vector<float> ref = readDataFromFile(refFilename);

    size_t queryLength = data.size() / BATCH_SIZE;

    float *query = data.data();
    float *hostRef = ref.data();
    float *deviceQuery, *deviceRef, *deviceScores, *deviceMinScores;
    size_t sizeData = data.size() * sizeof(float);
    size_t sizeRef = REF_LENGTH * sizeof(float);

    wbCheck(hipMalloc((void **) &deviceQuery, sizeData));
    wbCheck(hipMemcpy(deviceQuery, query, sizeData, hipMemcpyHostToDevice));

    float *means, *stdDevs;
    size_t statsSize = BATCH_SIZE * sizeof(float);
    wbCheck(hipMalloc((void **) &means, statsSize));
    wbCheck(hipMalloc((void **) &stdDevs, statsSize));

    std::cout << "Starting Normalizer ..." << std::endl;

    std::cout << "Query length: " << queryLength << std::endl;

    // warmup
    hipLaunchKernelGGL(
        computeMeansAndStdDevs2, 
        dim3 (NUM_BATCHES,1,1), // Only 1 block needed
        dim3 (BLOCK_SIZE2,1,1), 
        0, 0, 
        deviceQuery, means, stdDevs, queryLength
    );

    // Warm up GPU by running the kernel twice

    hipLaunchKernelGGL(
        computeMeansAndStdDevs2, 
        dim3 (NUM_BATCHES,1,1),
        dim3 (BLOCK_SIZE2,1,1), 
        0, 0, 
        deviceQuery, means, stdDevs, queryLength
    );

    hipDeviceSynchronize();
    float totalTime = 0.0;
    float milliseconds = 0.0;
    hipEvent_t start, stop;
    
    // Measure the total time of the kernel over 10 iterations
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
    for (int i = 0; i < 10; ++i) {
        hipLaunchKernelGGL(
            computeMeansAndStdDevs2, 
            dim3 (NUM_BATCHES,1,1),
            dim3 (BLOCK_SIZE2,1,1), 
            0, 0, 
            deviceQuery, means, stdDevs, queryLength
        );
    }
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&totalTime, start, stop);

    std::cout << "Normalizer Total Time: " << totalTime << " ms" << std::endl;
    float normThroughput = BATCH_SIZE * queryLength * 10 / (totalTime * 1000000);
    std::cout << "Normalizer Throughput: " << normThroughput << " Giga-Samples per Second"<< std::endl;

    hipDeviceSynchronize();

    // Init sDTW
    wbCheck(hipMalloc((void **) &deviceRef, sizeRef));
    wbCheck(hipMemcpy(deviceRef, hostRef, sizeRef, hipMemcpyHostToDevice));
    wbCheck(hipMalloc((void **) &deviceScores, REF_LENGTH * BATCH_SIZE * sizeof(float)));
    wbCheck(hipMalloc((void **) &deviceMinScores, statsSize));
    wbCheck(hipMemcpyToSymbol(HIP_SYMBOL(cQ), deviceQuery, sizeData, 0, hipMemcpyHostToDevice));

    std::cout << "Starting sDTW..." << std::endl;
    dim3 threadsPerBlock(p); // Adjust as needed
    dim3 numBlocks(BATCH_SIZE);

    // Warmup
    hipLaunchKernelGGL(
        dtw, 
        numBlocks, 
        threadsPerBlock, 
        0, 0, 
        deviceRef, deviceScores, deviceMinScores, 2000
    );
    
    hipLaunchKernelGGL(
        dtw, 
        numBlocks, 
        threadsPerBlock, 
        0, 0, 
        deviceRef, deviceScores, deviceMinScores, queryLength
    );

    hipDeviceSynchronize();
    totalTime = 0.0;
    hipEvent_t sDTWStart, sDTWStop;
    
    // Measure the total time of the kernel over 10 iterations
    hipEventCreate(&sDTWStart);
    hipEventCreate(&sDTWStop);
    hipEventRecord(sDTWStart, 0);
    for (int i = 0; i < 10; ++i) {
        hipLaunchKernelGGL(
            dtw, 
            numBlocks, 
            threadsPerBlock, 
            0, 0, 
            deviceRef, deviceScores, deviceMinScores, queryLength
        );
    }
    hipEventRecord(sDTWStop, 0);
    hipEventSynchronize(sDTWStop);
    hipEventElapsedTime(&totalTime, sDTWStart, sDTWStop);

    std::cout << "sDTW Total Time: " << totalTime << " ms" << std::endl;
    float sDTWThroughput = BATCH_SIZE * queryLength * 10 / (totalTime * 1000000);
    std::cout << "sDTW Throughput: " << sDTWThroughput << " Giga-Samples per Second" << std::endl;

    // Skipping copying back to CPU. Unnecessary for throughput test.

    wbCheck(hipFree(deviceQuery));
    wbCheck(hipFree(means));
    wbCheck(hipFree(stdDevs));
    wbCheck(hipFree(deviceRef));
    wbCheck(hipFree(deviceScores));
    wbCheck(hipFree(deviceMinScores));

    return 0;
}
