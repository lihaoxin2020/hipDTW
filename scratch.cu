// Kernel to calculate partial sums and squares for mean and std_dev calculation
__global__ void calcPartialSums(float *input, float *partialSums, float *partialSquares, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    __shared__ float cacheSum[BLOCK_SIZE];
    __shared__ float cacheSquare[BLOCK_SIZE];

    float tempSum = 0;
    float tempSquare = 0;
    while (idx < N) {
        tempSum += input[idx];
        tempSquare += input[idx] * input[idx];
        idx += stride;
    }

    cacheSum[threadIdx.x] = tempSum;
    cacheSquare[threadIdx.x] = tempSquare;

    __syncthreads();

    // Reduction within a block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cacheSum[threadIdx.x] += cacheSum[threadIdx.x + i];
            cacheSquare[threadIdx.x] += cacheSquare[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Write result for this block to global mem
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = cacheSum[0];
        partialSquares[blockIdx.x] = cacheSquare[0];
    }
}
