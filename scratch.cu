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

__global__ void sDTW(float* input, float* ref, int input_len, int ref_len, float* scores) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;


    float s[BATCH_SIZE][input_len][ref_len];
    if (idx_x < ref_len && idx_y < BATCH_SIZE) {
        int batch_offset = idx_y * input_len;
        // int qeury_pointer = query_offset + idx_x;
        if (idx_x == 0) {
            s[idx_y][0][0] = (input[batch_offset] - ref[0]) * (input[batch_offset] - ref[0]);
        }
        // init rest of first row. 0 init as cpu implementation
        if (idx_x > 0) {
            s[idx_y][0][idx_x] = 0;
        }

        int offset = 0 - idx_x;
        for (int i = 1; i < input_len + ref_len - 1; i++) {
            int row = offset + i;
            if (row > 0 && row < input_len) {
                int qeury_pointer = row + batch_offset;
                if (idx_x == 0) {
                    s[idx_y][row][0] = s[idx_y][qeury_pointer-1][0] + (input[qeury_pointer] - ref[0]) * (input[qeury_pointer] - ref[0]);
                } else {
                    s[idx_y][qeury_pointer][idx_x] = 
                        fmin(s[idx_y][qeury_pointer-1][idx_x], fmin(s[idx_y][qeury_pointer][idx_x-1], s[idx_y][qeury_pointer-1][idx_x-1])) + 
                        (input[qeury_pointer] - ref[idx_x]) * (input[qeury_pointer] - ref[idx_x]);
                }
            }
        }
        if (idx_x == 0) {
            float min_dist = INF;
            for (int j = 0; j < ref_len; ++j) {
                min_dist = fminf(min_dist, s[idx_y][input_len-1][j]);
            }
            scores[idx_y] = min_dist;
        }
    }
}

// sDTW in batch of 512 queries
__global__ void sDTW(float* input, float* ref, int input_len, int ref_len, float* score_metrics, float* scores) {
    // Calculate global thread index
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("%d %d", idx_x, idx_y);
    
    // Declare pointer for shared memory
    float *s = score_metrics;

    // Compute the 1D index for the 3D array emulation
    auto index3D = [=](int b, int i, int j) {
        return b * input_len * ref_len + i * ref_len + j;
    };

    if (idx_x < ref_len && idx_y < BATCH_SIZE) {
        printf("%d %d", idx_x, idx_y);
        int batch_offset = idx_y * input_len;
        if (idx_x == 0) {
            s[index3D(idx_y, 0, 0)] = (input[batch_offset] - ref[0]) * (input[batch_offset] - ref[0]);
        }

        // Initialize the rest of the first row
        if (idx_x > 0) {
            s[index3D(idx_y, 0, idx_x)] = 0;
        }

        __syncthreads(); // Synchronize to ensure the first row is initialized

        int offset = 0 - idx_x;
        for (int i = 1; i < input_len + ref_len - 1; i++) {
            int row = offset + i;
            if (row > 0 && row < input_len) {
                int query_pointer = row + batch_offset;
                if (idx_x == 0) {
                    s[index3D(idx_y, row, 0)] = s[index3D(idx_y, query_pointer-1, 0)] + (input[query_pointer] - ref[0]) * (input[query_pointer] - ref[0]);
                } else {
                    s[index3D(idx_y, query_pointer, idx_x)] =
                        fmin(s[index3D(idx_y, query_pointer-1, idx_x)], fmin(s[index3D(idx_y, query_pointer, idx_x-1)], s[index3D(idx_y, query_pointer-1, idx_x-1)])) + 
                        (input[query_pointer] - ref[idx_x]) * (input[query_pointer] - ref[idx_x]);
                }
            }
        }

        __syncthreads();

        // Compute the minimum distance for the last row
        if (idx_x == 0) {
            float min_dist = INF;
            for (int j = 0; j < ref_len; ++j) {
                min_dist = fminf(min_dist, s[index3D(idx_y, input_len-1, j)]);
            }
            scores[idx_y] = min_dist;
        }
    }
}
