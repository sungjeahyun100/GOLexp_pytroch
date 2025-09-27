#include "d_matrix_2.hpp"

namespace d_matrix_ver2 {
    // cuRAND state initialization kernel implementation
    __global__ void initCurandStates(curandState *states, unsigned long long seed, int total) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;
        curand_init(seed, idx, 0, &states[idx]);
    }

    // ========== 기본 매트릭스 연산 커널들 ==========
    
    template<typename T>
    __global__ void TransInKernel(T* d_A, T* d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
        if (x < row && y < col) {
            d_C[y * row + x] = d_A[x*col+y];
        }
    }
    
    template<typename T>
    __global__ void rotateInKernel(T *d_A, T *d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
        if (x < row && y < col) {
            int src_idx = x * col + y;
            int dst_idx = (row - 1 - x) * col + (col - 1 - y);
            d_C[dst_idx] = d_A[src_idx];
        }
    }
    
    template<typename T>
    __global__ void extract_batch(T* d_A, T* d_C, int sample_size, int begin_idx, int end_idx) {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row_idx < (end_idx - begin_idx)) {
            for (int col_idx = 0; col_idx < sample_size; ++col_idx) {
                int src_idx = (begin_idx + row_idx) * sample_size + col_idx;
                int dst_idx = row_idx * sample_size + col_idx;
                d_C[dst_idx] = d_A[src_idx];
            }
        }
    }

    template<typename T>
    __global__ void reduceRows(const T* d_A, T* d_C, int rows, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < cols) {
            T sum = T(0);
            for (int i = 0; i < rows; ++i) {
                sum += d_A[i * cols + x];
            }
            d_C[x] = sum/rows;
        }
    }

    // ========== 멤버 함수 구현들 ==========
    
    template<typename T>
    d_matrix_2<T> d_matrix_2<T>::transpose(cudaStream_t str) const {
        d_matrix_2<T> transposed(col, row);
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
        if(str == 0){
            TransInKernel<<<gridSize, blockSize>>>(d_data, transposed.getDevPointer(), row, col);
            CHECK_CUDA(cudaDeviceSynchronize());
        }else {
            TransInKernel<<<gridSize, blockSize, 0, str>>>(d_data, transposed.getDevPointer(), row, col);
            CHECK_CUDA(cudaStreamSynchronize(str));
        }
        return transposed;
    }
    
    template<typename T>
    d_matrix_2<T> d_matrix_2<T>::rotated180(cudaStream_t str) const {
        d_matrix_2<T> rotated(row, col);
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
        if(str == 0){
            rotateInKernel<<<gridSize, blockSize>>>(d_data, rotated.getDevPointer(), row, col);
            CHECK_CUDA(cudaDeviceSynchronize());
        }else {
            rotateInKernel<<<gridSize, blockSize, 0, str>>>(d_data, rotated.getDevPointer(), row, col);
            CHECK_CUDA(cudaStreamSynchronize(str));
        }
        return rotated;
    }

    template<typename T>
    d_matrix_2<T> d_matrix_2<T>::getBatch(int batchSize, int begin_idx, cudaStream_t str) {
        int sample_size = col;
        d_matrix_2<T> result(batchSize, sample_size);
        int threads = 256;
        int blocks  = (row + threads - 1)/threads;
        if(str == 0){
            extract_batch<<<blocks, threads>>>(d_data, result.getDevPointer(), sample_size, begin_idx, begin_idx+batchSize);
            CHECK_CUDA(cudaDeviceSynchronize());
        }else {
            extract_batch<<<blocks, threads, 0, str>>>(d_data, result.getDevPointer(), sample_size, begin_idx, begin_idx+batchSize);
            CHECK_CUDA(cudaStreamSynchronize(str));
        }
        return result;
    }

    // ========== 매트릭스 연산 커널들 ==========
    
    template<typename T>
    __global__ void zeroPad(const T* d_A, T* d_C, int row, int col, int pad_row, int pad_col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= pad_row || y >= pad_col) return;
        
        int idx_C = x * pad_col + y;
        
        // Calculate padding size (assuming symmetric padding)
        int pad_size = (pad_row - row) / 2;
        
        // Check if current position is in padding area
        if (x < pad_size || x >= pad_row - pad_size || y < pad_size || y >= pad_col - pad_size) {
            d_C[idx_C] = T(0);
        } else {
            // Map to input index
            int src_i = x - pad_size;
            int src_j = y - pad_size;
            int idx_A = src_i * col + src_j;
            d_C[idx_C] = d_A[idx_A];
        }
    }

    template<typename T>
    d_matrix_2<T> zeroPedding(const d_matrix_2<T> &d_A, int size, cudaStream_t str) {
        d_matrix_2<T> C(d_A.getRow()+(size*2), d_A.getCol()+(size*2));
        dim3 blockSize(32, 32);
        dim3 gridSize((C.getRow() + blockSize.x - 1) / blockSize.x, (C.getCol() + blockSize.y - 1) / blockSize.y);
        if(str == 0){
            zeroPad<<<gridSize, blockSize>>>(d_A.getDevPointer(), C.getDevPointer(), d_A.getRow(), d_A.getCol(), C.getRow(), C.getCol());
            cudaDeviceSynchronize();
        }else {
            zeroPad<<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), C.getDevPointer(), d_A.getRow(), d_A.getCol(), C.getRow(), C.getCol());
            cudaStreamSynchronize(str);
        }
        return C;
    }

    template<typename T>
    __global__ void HPinKernel_1dx(const T* __restrict__ d_A, const T* __restrict__ d_B, T* __restrict__ C, int row, int col){
        int idx = blockDim.x*blockIdx.x+threadIdx.x;
        if(idx >= row*col) return;
        C[idx] = d_A[idx]*d_B[idx];
    }
    
    template<typename T>
    __global__ void HPinKernel(
        const T* __restrict__ d_A,
        const T* __restrict__ d_B,
              T* __restrict__ d_C,
        int rows,
        int cols
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (row < rows && col < cols) {
            int idx = row * cols + col;
            d_C[idx] = d_A[idx] * d_B[idx];
            __syncthreads();
        }
    }

    template<typename T>
    d_matrix_2<T> HadamardProduct(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, cudaStream_t str) {
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        if(str == 0) {
            HPinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col); 
            cudaDeviceSynchronize();
        }else {
            HPinKernel<<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col); 
            cudaStreamSynchronize(str);
        }
        return C;
    }

    template<typename T>
    __global__ void ScalainKernel(T* d_A, T scalar, T* d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < row && y < col) {
            d_C[x * col + y] = d_A[x * col + y] * scalar;
        }
    }
    
    template<typename T>
    d_matrix_2<T> ScalaProduct(const d_matrix_2<T>& d_A, T scalar, cudaStream_t str) {
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        if(str == 0) {
            ScalainKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), scalar, C.getDevPointer(), row, col);
            cudaDeviceSynchronize();
        }else{
            ScalainKernel<<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), scalar, C.getDevPointer(), row, col);
            cudaStreamSynchronize(str);
        }
        return C;
    }

    template<typename T>
    __global__ void MPinKernel(T* d_A, T* d_B, T* d_C, int row, int col, int eq) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // row index
        int y = blockIdx.y * blockDim.y + threadIdx.y; // col index
    
        if (x < row && y < col) {
            T sum = 0;
            for (int i = 0; i < eq; i++) {
                sum += d_A[x * eq + i] * d_B[i * col + y];
            }
            d_C[x * col + y] = sum;
        }
    }
    
    template<typename T>
    d_matrix_2<T> matrixMP(const d_matrix_2<T>& A, const d_matrix_2<T>& B, cudaStream_t str) {
        int M = A.getRow();
        int N = B.getCol();
        int K = A.getCol();
    
        d_matrix_2<T> C(M, N);
    
        constexpr int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
        if(str == 0) {
            MPinKernel<T><<<grid, block>>>(A.getDevPointer(), B.getDevPointer(), C.getDevPointer(), M, N, K);
            cudaDeviceSynchronize();
        }
        else {
            MPinKernel<T><<<grid, block, 0, str>>>(A.getDevPointer(), B.getDevPointer(), C.getDevPointer(), M, N, K);
            cudaStreamSynchronize(str);
        }
        return C;
    }

    template<typename T>
    __global__ void PlusinKernel(T* d_A, T* d_B, T* d_C, int row, int col) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < row && y < col) {
            d_C[x * col + y] = d_A[x * col + y] + d_B[x * col + y]; 
        }
    }
    
    template<typename T>
    d_matrix_2<T> matrixPlus(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, cudaStream_t str){
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        if(str == 0) {
            PlusinKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
            cudaDeviceSynchronize();
        }else{
            PlusinKernel<<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), row, col);
            cudaStreamSynchronize(str);
        }
        return C;
    }

    // ========== 활성화 함수 관련 ==========
    
    template<typename T, T (*ActivateFunc)(T)>
    __global__ void ActivateInKernel(T* d_A, T* d_C, int row, int col){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < row && y < col) {
            d_C[x * col + y] = ActivateFunc(d_A[x * col + y]);
        }
    }
     
    template<typename T, T (*ActivateFunc)(T)>
    d_matrix_2<T> MatrixActivate(const d_matrix_2<T>& d_A, cudaStream_t str){
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);

        if(str == 0){
            ActivateInKernel<T, ActivateFunc><<<gridSize, blockSize>>>(d_A.getDevPointer(), C.getDevPointer(), row, col);
            cudaDeviceSynchronize();
        }else{
            ActivateInKernel<T, ActivateFunc><<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), C.getDevPointer(), row, col);
            cudaStreamSynchronize(str);
        }
        return C;
    }

    // ========== 스칼라 연산 ==========
    
    template<typename T>
    T plusAllElements(const d_matrix_2<T>& d_A){
        T result = 0.0;
        int row = d_A.getRow();
        int col = d_A.getCol();
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                result += d_A(i, j);
            }
        }
        return result;
    }
    
    template<typename T>
    __global__ void plusScalaToMatrix(T* d_C,int row, int col, T scala){
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
        if(idx_x < row && idx_y < col){
            d_C[idx_x * col + idx_y] += scala;
        }
    }
    
    template<typename T>
    d_matrix_2<T> ScalaPlus(const d_matrix_2<T>& d_A, T scala){
        int row = d_A.getRow();
        int col = d_A.getCol();
    
        d_matrix_2<T> C(row, col);
        C = d_A;
    
        dim3 blockSize(32, 32);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x, (col + blockSize.y - 1) / blockSize.y);
    
        plusScalaToMatrix<<<gridSize, blockSize>>>(C.getDevPointer(), row, col, scala);
        cudaDeviceSynchronize();
    
        return C;
    }

    // ========== 타입 캐스팅 ==========
    
    template <typename T>
    __global__ void castKernel(const T* src, double* dst, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            dst[idx] = static_cast<double>(src[idx]);
        }
    }
    
    template <typename T>
    d_matrix_2<double> castToDoubleGPU(const d_matrix_2<T>& input) {
        d_matrix_2<double> output(input.getRow(), input.getCol());
    
        T* d_src = input.getDevPointer();
        double* d_dst = output.getDevPointer();
    
        int size = input.getRow()*input.getCol();
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
    
        castKernel<<<numBlocks, blockSize>>>(d_src, d_dst, size);
        cudaDeviceSynchronize();
    
        return output;
    }

    // ========== Softmax 구현 ==========
    
    template<typename T>
    __global__ void softmaxKernel(T* in, T* out, int row, int col) {
        int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (rowIdx >= row) return;
    
        // 1. max
        double max_val = in[rowIdx * col + 0];
        for (int i = 0; i < col; ++i) {
            double val = in[rowIdx * col + i];
            if (val > max_val) max_val = val;
        }
    
        // 2. 분자와 분모
        double sum = 0.0;
        for (int i = 0; i < col; ++i) {
            sum += exp(in[rowIdx * col + i] - max_val);
        }
    
        // 3. 결과 저장
        for (int i = 0; i < col; ++i) {
            out[rowIdx * col + i] = exp(in[rowIdx * col + i] - max_val) / sum;
        }
    }

    template<typename T>
    d_matrix_2<T> softmax(const d_matrix_2<T>& input, cudaStream_t str) {
        int row = input.getRow();
        int col = input.getCol();
    
        d_matrix_2<T> output(row, col);
    
        int threads = 32;
        int blocks = (row*col + threads - 1) / threads;
    
        if(str == 0){
            softmaxKernel<<<blocks, threads>>>(
                input.getDevPointer(),
                output.getDevPointer(),
                row, col
            );
            cudaDeviceSynchronize();
        }else {
            softmaxKernel<<<blocks, threads, 0, str>>>(
                input.getDevPointer(),
                output.getDevPointer(),
                row, col
            );
            cudaStreamSynchronize(str);
        }
        return output;
    }

    template<typename T>
    __global__ void efficientSoftmaxKernel(const T* in, T* out, int rows, int cols) {
        int row = blockIdx.x;
        if (row >= rows) return;
    
        int tid = threadIdx.x;
        const int block_size = blockDim.x;
    
        // Use dynamic shared memory with byte addressing
        extern __shared__ char shared_mem[];
        T* s_cache = reinterpret_cast<T*>(shared_mem);
    
        // --- 1. 행의 최댓값 찾기 ---
        T thread_max = -1.0/0.0; // -INFINITY
        for (int i = tid; i < cols; i += block_size) {
            thread_max = max(thread_max, in[row * cols + i]);
        }
        s_cache[tid] = thread_max;
        __syncthreads();
    
        // block_size가 2의 거듭제곱이라고 가정
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_cache[tid] = max(s_cache[tid], s_cache[tid + s]);
            }
            __syncthreads();
        }
        const T max_val = s_cache[0];
    
        // --- 2. exp(x - max)의 합계 구하기 ---
        T thread_sum = 0.0;
        for (int i = tid; i < cols; i += block_size) {
            thread_sum += exp(in[row * cols + i] - max_val);
        }
        s_cache[tid] = thread_sum;
        __syncthreads();
    
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_cache[tid] += s_cache[tid + s];
            }
            __syncthreads();
        }
        const T sum_val = s_cache[0];
    
        // --- 3. 최종 결과 계산 ---
        for (int i = tid; i < cols; i += block_size) {
            out[row * cols + i] = exp(in[row * cols + i] - max_val) / (sum_val + 1e-9);
        }
    }
    
    template<typename T>
    d_matrix_2<T> softmax_efficient(const d_matrix_2<T>& input, cudaStream_t str) {
        int row = input.getRow();
        int col = input.getCol();
    
        d_matrix_2<T> output(row, col);

        int threads = 512;
        int blocks = row;
    
        if(str == 0){
            efficientSoftmaxKernel<T><<<blocks, threads, threads*sizeof(T)>>>(input.getDevPointer(), output.getDevPointer(), row, col);
            cudaDeviceSynchronize();
        }else {
            efficientSoftmaxKernel<T><<<blocks, threads, threads*sizeof(T), str>>>(input.getDevPointer(), output.getDevPointer(), row, col);
            cudaStreamSynchronize(str);
        }
        return output;
    }

    // ========== 기타 유틸리티 함수들 ==========
    
    template<typename T>
    __global__ void convertInKernel(T* d_X, int row, int col){
        double epsilon = 1e-10;
        int idx_x = blockDim.x*blockIdx.x+threadIdx.x;
        int idx_y = blockDim.y*blockIdx.y+threadIdx.y;
    
        if(idx_y < row && idx_x < col){
            if(d_X[idx_y*col+idx_x] < epsilon){
                d_X[idx_y*col+idx_x] = epsilon;
            }
        }
    }
    
    template<typename T>
    d_matrix_2<T> convertZeroToEpsilon(d_matrix_2<T> x, cudaStream_t str){
        int row = x.getRow();
        int col = x.getCol();
    
        dim3 blockSize(32, 32);
        dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    
        if(str == 0){
            convertInKernel<<<gridSize, blockSize>>>(x.getDevPointer(), row, col);
            cudaDeviceSynchronize();
        }else {
            convertInKernel<<<gridSize, blockSize, 0, str>>>(x.getDevPointer(), row, col);
            cudaStreamSynchronize(str);
        }
        return x;
    }

    // ========== 컨볼루션 연산 ==========
    
    template<typename T>
    __global__ void convoluteInKernel(T* __restrict__ d_A, T* __restrict__ d_B, T* __restrict__ d_C, int inputRow, int inputCol, int filterRow, int filterCol, int outputRow, int outputCol, int stride) {
        int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
        int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
        if (idx_x < outputRow && idx_y < outputCol) {
            T sum = 0;
            for (int i = 0; i < filterRow; i++) {
                for (int j = 0; j < filterCol; j++) {
                    int input_x = idx_x*stride + i;
                    int input_y = idx_y*stride + j;
                    if (input_x < inputRow && input_y < inputCol) {
                        sum += d_A[input_x * inputCol + input_y] * d_B[i * filterCol + j];
                    }
                }
            }
            d_C[idx_x * outputCol + idx_y] = sum;
        }
    }
    
    template<typename T>
    d_matrix_2<T> convolute(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, int stride, cudaStream_t str) {
        int inputRow = d_A.getRow();
        int inputCol = d_A.getCol();
        int filterRow = d_B.getRow();
        int filterCol = d_B.getCol();
    
        int outputRow = ((inputRow - filterRow)/stride) + 1;
        int outputCol = ((inputCol - filterCol)/stride) + 1;
    
        d_matrix_2<T> C(outputRow, outputCol);
    
        dim3 blockSize(32, 32);
        dim3 gridSize((outputRow + blockSize.x - 1) / blockSize.x, (outputCol + blockSize.y - 1) / blockSize.y);
    
        if(str == 0){
            convoluteInKernel<<<gridSize, blockSize>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), inputRow, inputCol, filterRow, filterCol, outputRow, outputCol, stride);
            cudaDeviceSynchronize();
        }else {
            convoluteInKernel<<<gridSize, blockSize, 0, str>>>(d_A.getDevPointer(), d_B.getDevPointer(), C.getDevPointer(), inputRow, inputCol, filterRow, filterCol, outputRow, outputCol, stride);
            cudaStreamSynchronize(str);
        }
        return C;
    }

    // ========== 가중치 초기화 ==========
    
    template<typename T>
    __global__ void InitWeightInKernel(
        T*               d_weight,
        curandState*     states,
        int              row,
        int              col,
        InitType         type
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= row || y >= col) return;
    
        int idx = x * col + y;
        curandState localState = states[idx];
    
        T val;
        switch (type) {
            case InitType::He: {
                val = curand_normal(&localState) * sqrt(2.0 / (double)row);
                break;
            }
            case InitType::Xavier: {
                val = curand_normal(&localState) * sqrt(1.0 / (double)row);
                break;
            }
            case InitType::Uniform: {
                val = (curand_uniform(&localState) * 2.0) - 1.0;
                break;
            }
        }
        d_weight[idx] = val;
        states[idx] = localState;
    }
    
    template<typename T>
    d_matrix_2<T> InitWeight(int row, int col, InitType type, cudaStream_t str) {
        d_matrix_2<T> weight(row, col);
    
        int total = row * col;
        curandState *d_states;
        cudaMalloc(&d_states, sizeof(curandState) * total);
    
        std::random_device rd;
        std::mt19937_64 mt(rd());
        std::uniform_int_distribution<unsigned long long> dist;
        unsigned long long seed = dist(mt);
    
        int threadsInit = 256;
        int blocksInit = (total + threadsInit - 1) / threadsInit;
        if(str == 0){
            initCurandStates<<<blocksInit, threadsInit>>>(d_states, seed, total);
            cudaDeviceSynchronize();
        }else {
            initCurandStates<<<blocksInit, threadsInit, 0, str>>>(d_states, seed, total);
            cudaStreamSynchronize(str);
        }
    
        dim3 blockSize(16, 16);
        dim3 gridSize((row + blockSize.x - 1) / blockSize.x,
                      (col + blockSize.y - 1) / blockSize.y);
        if(str == 0){
            InitWeightInKernel<<<gridSize, blockSize>>>(
                weight.getDevPointer(),
                d_states,
                row,
                col,
                type
            );
            cudaDeviceSynchronize();
        }else {
            InitWeightInKernel<<<gridSize, blockSize, 0, str>>>(
                weight.getDevPointer(),
                d_states,
                row,
                col,
                type
            );
            cudaStreamSynchronize(str);
        }
    
        cudaFree(d_states);
        return weight;
    }

    // ========== 매트릭스 연결 ==========
    
    template<typename T>
    d_matrix_2<T> concatenate(const d_matrix_2<T>& A, const d_matrix_2<T>& B, cudaStream_t str) {
        if (A.getRow() != B.getRow()) {
            throw std::runtime_error("Matrices must have the same number of rows for concatenation");
        }
        
        int rows = A.getRow();
        int cols_A = A.getCol();
        int cols_B = B.getCol();
        int total_cols = cols_A + cols_B;
        
        d_matrix_2<T> result(rows, total_cols, str);
        
        cudaMemcpy2DAsync(
            result.getDevPointer(),
            total_cols * sizeof(T),
            A.getDevPointer(),
            cols_A * sizeof(T),
            cols_A * sizeof(T),
            rows,
            cudaMemcpyDeviceToDevice,
            str
        );
        
        cudaMemcpy2DAsync(
            result.getDevPointer() + cols_A,
            total_cols * sizeof(T),
            B.getDevPointer(),
            cols_B * sizeof(T),
            cols_B * sizeof(T),
            rows,
            cudaMemcpyDeviceToDevice,
            str
        );
        
        return result;
    }

    // ========== 템플릿 인스턴스화 ==========
    
    // d_matrix_2 클래스
    template class d_matrix_2<double>;
    template class d_matrix_2<float>;
    
    // 자주 사용되는 타입들에 대한 명시적 인스턴스화
    template d_matrix_2<double> zeroPedding(const d_matrix_2<double>&, int, cudaStream_t);
    template d_matrix_2<float> zeroPedding(const d_matrix_2<float>&, int, cudaStream_t);
    
    template d_matrix_2<double> HadamardProduct(const d_matrix_2<double>&, const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> HadamardProduct(const d_matrix_2<float>&, const d_matrix_2<float>&, cudaStream_t);
    
    template d_matrix_2<double> ScalaProduct(const d_matrix_2<double>&, double, cudaStream_t);
    template d_matrix_2<float> ScalaProduct(const d_matrix_2<float>&, float, cudaStream_t);
    
    template d_matrix_2<double> matrixMP(const d_matrix_2<double>&, const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> matrixMP(const d_matrix_2<float>&, const d_matrix_2<float>&, cudaStream_t);
    
    template d_matrix_2<double> matrixPlus(const d_matrix_2<double>&, const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> matrixPlus(const d_matrix_2<float>&, const d_matrix_2<float>&, cudaStream_t);
    
    template d_matrix_2<double> softmax(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> softmax(const d_matrix_2<float>&, cudaStream_t);
    
    template d_matrix_2<double> softmax_efficient(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> softmax_efficient(const d_matrix_2<float>&, cudaStream_t);
    
    template d_matrix_2<double> convertZeroToEpsilon(d_matrix_2<double>, cudaStream_t);
    template d_matrix_2<float> convertZeroToEpsilon(d_matrix_2<float>, cudaStream_t);
    
    template d_matrix_2<double> convolute(const d_matrix_2<double>&, const d_matrix_2<double>&, int, cudaStream_t);
    template d_matrix_2<float> convolute(const d_matrix_2<float>&, const d_matrix_2<float>&, int, cudaStream_t);
    
    template d_matrix_2<double> InitWeight(int, int, InitType, cudaStream_t);
    template d_matrix_2<float> InitWeight(int, int, InitType, cudaStream_t);
    
    template d_matrix_2<double> concatenate(const d_matrix_2<double>&, const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<float> concatenate(const d_matrix_2<float>&, const d_matrix_2<float>&, cudaStream_t);

    // 활성화 함수들의 명시적 템플릿 인스턴스화
    template d_matrix_2<double> MatrixActivate<double, &relu<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &lrelu<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &sigmoid<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &Identity<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &Tanh<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &ELU<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &SELU<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &Softplus<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &Softsign<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &Swish<double>>(const d_matrix_2<double>&, cudaStream_t);
    
    // 활성화 함수 미분들
    template d_matrix_2<double> MatrixActivate<double, &d_relu<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_lrelu<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_sigmoid<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_I<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_tanh<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_ELU<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_SELU<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_Softsign<double>>(const d_matrix_2<double>&, cudaStream_t);
    template d_matrix_2<double> MatrixActivate<double, &d_Swish<double>>(const d_matrix_2<double>&, cudaStream_t);
    
    // Kernel functions need template instantiation for double type
    template __global__ void MPinKernel<double>(double*, double*, double*, int, int, int);
    template __global__ void HPinKernel_1dx<double>(const double*, const double*, double*, int, int);
    template __global__ void reduceRows<double>(const double*, double*, int, int);
    template __global__ void TransInKernel<double>(double*, double*, int, int);
    template __global__ void ScalainKernel<double>(double*, double, double*, int, int);

}
