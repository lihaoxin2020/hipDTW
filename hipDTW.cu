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

// #define BLOCK_SIZE 512 // Adjust based on your GPU's architecture
#define BATCH_SIZE 512 // Normalizer batch size
#define INF 1e20
#define k 1563
#define p 64
#define REF_LENGTH 100000

#define BLOCK_SIZE2 1000
#define NUM_BATCHES 512

__constant__ float cQ[2000 * BATCH_SIZE];

__global__ void dtw(float *subjects, float *scores, float *min_scores, int m) {
    float S[k]; // Registers storing subject data
    float M[k]; // Registers storing DP matrix cells
    float M_left, M_diag; // Registers for data from left neighbor
    float Reg_Q0, Reg_Q1; // Registers storing query data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("t %zu \n", (size_t)idx);

    // load subjects
    for (int i = 0; i < k; i++) {
        S[i] = subjects[threadIdx.x * k + i];
    }
    // init DP matrix
    // if (threadIdx.x % p == 0) {
    //     // M_left = (S[0] - cQ[0]) * (S[0] - cQ[0]);
    //     M[0] = 0; M_diag = 0; M_left = INF;
    // }
    for (int i = 0; i < k; i++) {
        if (threadIdx.x % p == 0) M[i] = INF;
        else M[i] = 0;
    }
    M_diag = 0; M_left = INF;
    // Load one query value per thread
    Reg_Q1 = cQ[blockIdx.x * m + threadIdx.x % p];
    
    // if(idx < 201) printf("t %zu %f \n", (size_t)threadIdx.x, Reg_Q1);
    if (threadIdx.x % p == 0) {
        Reg_Q0 = Reg_Q1;
    }
    else Reg_Q0 = INF;
    Reg_Q1 = __shfl_down(Reg_Q1, 1);
    
    for (int i = 1; i <= m+p; i++) { // Wavefront loop
        // Compute DP cells per thread using registers only
        float upper;
        for (int j = 0; j < k; j++) {
            // float temp = M[j];
            float temp;
            if (i - threadIdx.x == 1) temp = (S[j] - Reg_Q0) * (S[j] - Reg_Q0);
            else temp = fmin(M_left, fmin(M_diag, M[j])) + (S[j] - Reg_Q0) * (S[j] - Reg_Q0);
            // if (i == idx * k + j+1 + idx) {
            // if (i < 4 && (idx == 1 || idx == 0)) {
            // if (idx == 2 && i < 5) {
            //     printf("iter %d, thread %d, k %d, M_left %f, M_diag %f, upper %f, Q %f, R %f, res %f\n", i, idx, j, M_left, M_diag, M[j], Reg_Q0, S[j], temp);
            // }
            // if (temp < 0)
            // printf("iter %d, thread %d, k %d, score %f\n", i, idx, j, temp);
            M_diag = M[j];
            upper = M[j];
            M_left = temp;
            M[j] = temp;
        }
        // __syncthreads();
        // M_diag = __shfl_up(M[k-1], 1);
        M_diag = M_left;
        // if (idx == 0) {
        //     // for (int n = 0; n < k; n++)
        //     int n = i - 1;
        //     printf("Iter %i thread %d k %d %f \n", i, idx, n, M[n]);
        // }

        // Copy rightmost DP cell to neighboring thread
        __syncthreads();
        M_left = __shfl_up(M[k-1], 1);
        M_diag = __shfl_up(upper, 1);
        if (threadIdx.x % p == 0) {
            M_left = INF;
            M_diag = INF;
        }
        // M_diag = M_left;

        // Load new query data to register every p iterations
        if (i % p == 0 && i + threadIdx.x < 2000) Reg_Q1 = cQ[m * blockIdx.x + i + threadIdx.x % p];
        
        // Shuffle query registers
        // if (i <= m && idx == 63) printf("before t %zu, iter %d, Reg_Q0 %f, Reg_Q1 %f\n", (size_t) threadIdx.x, i, Reg_Q0, Reg_Q1);
        __syncthreads();
        Reg_Q0 = __shfl_up(Reg_Q0, 1);
        __syncthreads();
        if (threadIdx.x % p == 0) {
            // printf("iter %d, t %zu, Reg_Q0 %f, Reg_Q1 %f\n", i, (size_t) threadIdx.x, Reg_Q0, Reg_Q0);
            Reg_Q0 = Reg_Q1;
        }
        // Reg_Q0 = Reg_Q1;
        Reg_Q1 = __shfl_down(Reg_Q1, 1);
        __syncthreads();
        // if (threadIdx.x % p == 0) Reg_Q0 = Reg_Q1;
        // __syncthreads();
        // if (i <= m && idx == 63) printf("t %zu, iter %d, Reg_Q0 %f, Reg_Q1 %f\n", (size_t) threadIdx.x, i, Reg_Q0, Reg_Q1);
        
        // output_DTW_score(M[k-1], scores, threadIdx.x + blockIdx.x * k, k, p);
        if (i - threadIdx.x == m) {
            for (int j = 0; j < k && threadIdx.x * k + j < REF_LENGTH; j++) {
                scores[REF_LENGTH * blockIdx.x + threadIdx.x * k + j] = M[j];
                // if (blockIdx.x <= 2) 
                //     printf("block %zu, write %f to scores[%d]\n", (size_t) blockIdx.x, M[j], REF_LENGTH * blockIdx.x + threadIdx.x * k + j);
                // scores[idx * k + j] = 0.01;
                // if (idx * k + j == 1999) printf("j %d, score %f", j, M[j]);
            }
        }
    }
    // printf("end of for\n");
    if (threadIdx.x == 0) {
        float min_score = INF;
        for (int i = 0; i < 100000; i++) {
            min_score = fmin(scores[REF_LENGTH * blockIdx.x + i], min_score);
        }
        // printf("block %zu, score %f \n", (size_t) blockIdx.x, min_score);
        min_scores[blockIdx.x] = min_score;
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
    float *deviceQuery, *deviceRef, *deviceScores, *deviceMinScores;
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
    wbCheck(hipMalloc((void **) &deviceScores, 100000*BATCH_SIZE*sizeof(float)));
    wbCheck(hipMalloc((void **) &deviceMinScores, statsSize));
    // wbCheck(hipMalloc((void **) &scoreMetrics, statsSize * queryLength * refLength));

    std::cout << "Starting kernel..." << std::endl;

    printf("Query length: %zu\n", queryLength);

    hipLaunchKernelGGL(
        computeMeansAndStdDevs2, 
        dim3 (NUM_BATCHES,1,1), // Only 1 block needed
        dim3 (BLOCK_SIZE2,1,1), 
        0, 0, 
        deviceQuery, means, stdDevs, queryLength
    );
    
    std::cout << "Normalized. Starting sDTW..." << std::endl;
    dim3 threadsPerBlock(p); // Adjust as needed
    dim3 numBlocks(BATCH_SIZE);

    // size_t sharedMemSize = BATCH_SIZE * queryLength * refLength * sizeof(float);

    wbCheck(hipMemcpyToSymbol(HIP_SYMBOL(cQ), deviceQuery, sizeData, 0, hipMemcpyHostToDevice));

    std::cout << "starting sDTW..." << std::endl;
    hipLaunchKernelGGL(
        dtw, 
        numBlocks, 
        threadsPerBlock, 
        0, 0, 
        deviceRef, deviceScores, deviceMinScores, 2000
    );

    // Copy means stdDevs to host
    float *hostMeans = new float[BATCH_SIZE];
    float *hostStdDevs = new float[BATCH_SIZE];
    float *normalizedQueries = new float[data.size()];
    float *hostScores = new float[BATCH_SIZE * ref.size()];
    float *hostMinScores = new float[BATCH_SIZE];
    wbCheck(hipMemcpy(hostMeans, means, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostStdDevs, stdDevs, statsSize, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(normalizedQueries, deviceQuery, sizeData, hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostScores, deviceScores, BATCH_SIZE * ref.size()*sizeof(float), hipMemcpyDeviceToHost));
    wbCheck(hipMemcpy(hostMinScores, deviceMinScores, statsSize, hipMemcpyDeviceToHost));

    // for (int i = 0; i < BATCH_SIZE; i++) {
    //     std::cout << "Mean: " << hostMeans[i] << ", Standard Deviation: " << hostStdDevs[i] << std::endl;
    // }

    // for (int i = 0; i < ref.size(); i++) {
    //     std::cout << "element " << i << " sDTW scores: " << hostScores[i] << std::endl;
    // }
    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "Batch " << i << " sDTW scores: " << hostMinScores[i] << std::endl;
    }

    delete[] hostMeans;
    delete[] hostStdDevs;
    delete[] normalizedQueries;
    delete[] hostScores;
    delete[] hostMinScores;
    wbCheck(hipFree(deviceQuery));
    wbCheck(hipFree(means));
    wbCheck(hipFree(stdDevs));
    wbCheck(hipFree(deviceRef));
    wbCheck(hipFree(deviceScores));
    wbCheck(hipFree(deviceMinScores));

    return 0;
}
