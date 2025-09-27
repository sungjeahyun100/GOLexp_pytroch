#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>
#include <ostream>
#include <stdexcept>

#define CHECK_CUDA(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(cudaGetErrorString(err)); \
            } \
        } while (0)

namespace d_matrix_ver2{

    enum class InitType
    {
        He,
        Xavier,
        Uniform
    };

    // 커널 함수 선언들
    template<typename T>
    __global__ void TransInKernel(T* d_A, T* d_C, int row, int col);
    
    template<typename T>
    __global__ void rotateInKernel(T *d_A, T *d_C, int row, int col);
    
    template<typename T>
    __global__ void extract_batch(T* d_A, T* d_C, int sample_size, int begin_idx, int end_idx);

    template<typename T>
    __global__ void reduceRows(const T* d_A, T* d_C, int rows, int cols);

    // GPU 전용 행렬 클래스
    template<typename T>
    class d_matrix_2 {
    private:
        T* d_data;
        std::vector<T> h_data;
        int row, col;
        bool d_allocated;
        
    public:
        // 생성자들
        d_matrix_2(int r, int c, cudaStream_t str=0) : row(r), col(c) {
            size_t size = static_cast<size_t>(row) * col;
            if (size == 0) {
                throw std::invalid_argument("Matrix dimensions must be positive");
            }
            h_data.resize(size);
            CHECK_CUDA(cudaMallocAsync(&d_data, size * sizeof(T), str));
            CHECK_CUDA(cudaMemsetAsync(d_data, 0, size * sizeof(T), str)); // 초기화
            d_allocated = true;
        }
        
        d_matrix_2(cudaStream_t str=0) : row(1), col(1){
            h_data.resize(1);
            CHECK_CUDA(cudaMalloc(&d_data, 1 * sizeof(T)));
            d_allocated = true;
        }
        
        d_matrix_2(std::initializer_list<std::initializer_list<T>> list2d){
            row = list2d.size();
            if (row == 0) throw std::invalid_argument("Empty matrix");
            col = list2d.begin()->size();
            if (col == 0) throw std::invalid_argument("Empty row");
            
            for (auto& inner_list : list2d) {
                if (inner_list.size() != static_cast<size_t>(col)) {
                    throw std::invalid_argument("Inconsistent row sizes");
                }
            }
            
            size_t size = static_cast<size_t>(row) * col;
            h_data.resize(size);
            
            int idx = 0;
            for (auto& inner_list : list2d) {
                for (auto& val : inner_list) {
                    h_data[idx++] = val;
                }
            }
            
            CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(T)));
            d_allocated = true;
            cpyToDev();
        }
        
        // 복사 생성자
        d_matrix_2(const d_matrix_2<T>& other) : row(other.row), col(other.col) {
            size_t size = static_cast<size_t>(row) * col;
            h_data = other.h_data;
            CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(d_data, other.d_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
            d_allocated = true;
        }
        
        d_matrix_2<T>& operator=(const d_matrix_2<T>& other) {
            if (this != &other) {
                if (d_allocated) {
                    cudaFree(d_data);
                }
                row = other.row;
                col = other.col;
                h_data = other.h_data;
                size_t size = static_cast<size_t>(row) * col;
                CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(T)));
                CHECK_CUDA(cudaMemcpy(d_data, other.d_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
                d_allocated = true;
            }
            return *this;
        }
        
        // 이동 생성자
        d_matrix_2(d_matrix_2&& other) noexcept
            : d_data(other.d_data), h_data(std::move(other.h_data)),
              row(other.row), col(other.col), d_allocated(other.d_allocated) {
            other.d_data = nullptr;
            other.d_allocated = false;
            other.row = 0;
            other.col = 0;
        }
        
        d_matrix_2& operator=(d_matrix_2&& other) noexcept {
            if (this != &other) {
                if (d_allocated) {
                    cudaFree(d_data);
                }
                d_data = other.d_data;
                h_data = std::move(other.h_data);
                row = other.row;
                col = other.col;
                d_allocated = other.d_allocated;
                
                other.d_data = nullptr;
                other.d_allocated = false;
                other.row = 0;
                other.col = 0;
            }
            return *this;
        }
        
        // 소멸자
        ~d_matrix_2() {
            if (d_allocated) {
                cudaFree(d_data);
            }
        }
        
        // 기본 접근자들
        int getRow() const { return row; }
        int getCol() const { return col; }
        size_t size() const { return static_cast<size_t>(row) * col; }
        T* getDevPointer() const { return d_data; }
        const T* getHostPointer() const { return h_data.data(); }
        T* getHostPointer() { return h_data.data(); }
        
        // 호스트-디바이스 메모리 관리
        void cpyToHost(cudaStream_t str = 0) {
            size_t bytes = size() * sizeof(T);
            if (str == 0) {
                CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
            } else {
                CHECK_CUDA(cudaMemcpyAsync(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost, str));
            }
        }
        
        void cpyToDev(cudaStream_t str = 0) {
            size_t bytes = size() * sizeof(T);
            if (str == 0) {
                CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
            } else {
                CHECK_CUDA(cudaMemcpyAsync(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice, str));
            }
        }
        
        // 요소 접근
        T& operator()(int r, int c) {
            if (r >= row || c >= col || r < 0 || c < 0) {
                throw std::out_of_range("Index out of range");
            }
            return h_data[r * col + c];
        }
        
        const T& operator()(int r, int c) const {
            if (r >= row || c >= col || r < 0 || c < 0) {
                throw std::out_of_range("Index out of range");
            }
            return h_data[r * col + c];
        }
        
        // 새로 추가된 유틸리티 함수들
        void randomInit(T min_val, T max_val, cudaStream_t str = 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(static_cast<double>(min_val), static_cast<double>(max_val));
            
            for (size_t i = 0; i < size(); ++i) {
                h_data[i] = static_cast<T>(dis(gen));
            }
            cpyToDev(str);
        }

        void setHostData(std::vector<T>& in){
            h_data = in;
        }
        
        void setHostValue(int r, int c, T value) {
            if (r >= row || c >= col || r < 0 || c < 0) {
                throw std::out_of_range("Index out of range in setHostValue");
            }
            h_data[r * col + c] = value;
        }
        
        void setHostValueAndSync(int r, int c, T value, cudaStream_t str = 0) {
            setHostValue(r, c, value);
            cpyToDev(str);
        }
        
        T getHostValue(int r, int c) const {
            if (r >= row || c >= col || r < 0 || c < 0) {
                throw std::out_of_range("Index out of range in getHostValue");
            }
            return h_data[r * col + c];
        }

        std::vector<T> getHostData() const {
            return h_data;
        }
        
        void fill(T value, cudaStream_t str = 0) {
            std::fill(h_data.begin(), h_data.end(), value);
            cpyToDev(str);
        }
        
        void resize(int newR, int newC, cudaStream_t str = 0) {
            if (d_allocated) {
                cudaFree(d_data);
            }
            
            row = newR;
            col = newC;
            size_t new_size = static_cast<size_t>(row) * col;
            h_data.resize(new_size);
            
            CHECK_CUDA(cudaMalloc(&d_data, new_size * sizeof(T)));
            d_allocated = true;
        }

            
        
        void printMatrix() const {
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < col; ++j) {
                    std::cout << std::setw(4) << std::fixed << std::setprecision(3) 
                             << h_data[i * col + j];
                }
                std::cout << std::endl;
            }
        }
        
        // 반복자 지원
        T*       begin()       noexcept { return h_data.data(); }
        T*       end()         noexcept { return h_data.data() + row*col; }
        T const* begin() const noexcept { return h_data.data(); }
        T const* end()   const noexcept { return h_data.data() + row*col; }
    
        d_matrix_2<T> flatten(){
            d_matrix_2<T> V = *this;
            V.row = row*col;
            V.col = 1;
            return V;
        }
    
        d_matrix_2<T> reshape(int newR, int newC) const {
            if (static_cast<size_t>(newR) * newC != size())
                throw std::invalid_argument("reshape: size mismatch");
            d_matrix_2<T> V = *this;
            V.row = newR;
            V.col = newC;
            return V;
        }
    
        d_matrix_2<T> transpose(cudaStream_t str = 0) const;
        d_matrix_2<T> rotated180(cudaStream_t str = 0) const;
        d_matrix_2<T> getBatch(int batchSize, int begin_idx, cudaStream_t str = 0);
    };
    
    // 스트림 출력 연산자
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const d_matrix_2<T> &matrix)
    {
        os << matrix.getRow() << "x" << matrix.getCol() << "\n";
        for (int i = 0; i < matrix.getRow(); ++i) {
            for (int j = 0; j < matrix.getCol(); ++j) {
                os << matrix(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }
    
    // ========== 활성화 함수들 (디바이스 함수) ==========
    template<typename T>
    __device__ T relu(T x) { return x > 0 ? x : 0; }
     
    template<typename T>
    __device__ T d_relu(T x) { return x > 0 ? 1 : 0; }
    
    template<typename T>
    __device__ T lrelu(T x) { return x > 0 ? x : 0.01*x; }
    
    template<typename T>
    __device__ T d_lrelu(T x) { return x > 0 ? 1 : 0.01; }
    
    const double alpha = 1.6732632423543772848170429916717l;
    const double lamda = 1.0507009873554804934193349852946l;
    
    template<typename T>
    __device__ T ELU(T x) { return x > 0 ? x : alpha*(exp(x) - 1); }
    
    template<typename T>
    __device__ T d_ELU(T x) { return x > 0 ? 1 : alpha*exp(x); }
    
    template<typename T>
    __device__ T SELU(T x) { return x > 0 ? lamda*x : alpha*lamda*(exp(x) - 1); }
    
    template<typename T>
    __device__ T d_SELU(T x) { return x > 0 ? lamda : alpha*lamda*exp(x); }
    
    template <typename T>
    __device__ T Softplus(T x) {
        if (x > T(0)) {
            return x + log1p(exp(-x));
        } else {
            return log1p(exp(x));
        }
    }
    
    template <typename T>
    __device__ T Softsign(T x) { return x/(T(1)+fabs(x)); }
    
    template <typename T>
    __device__ T d_Softsign(T x) {
        T n = T(1)+fabs(x);
        return T(1)/(n*n);
    }
    
    template <typename T>
    __device__ T Swish(T x) { return x / (T(1) + exp(-x)); }
    
    template <typename T>
    __device__ T d_Swish(T x) {
        T ex   = exp(x);
        T denom = ex + T(1);
        return ex * (x + ex + 1.0) / (denom * denom); 
    }
    
    template<typename T>
    __device__ T Identity(T x) { return x; }
    
    template<typename T>
    __device__ T sigmoid(T x) { return T(1) / (T(1) + exp(-x)); }
    
    template<typename T>
    __device__ T d_sigmoid(T x) {
        T s = T(1) / (T(1) + exp(-x));
        return s * (1.0 - s);
    }
    
    template<typename T>
    __device__ T d_I(T x) { return T(1); }
    
    template<typename T>
    __device__ T Tanh(T x) { return tanh(x); }
    
    template<typename T>
    __device__ T d_tanh(T x) { return (1-tanh(x))*(1+tanh(x)); }
    
    template<typename T>
    __device__ T sqr(T x) { return sqrt(x); }
    
    template<typename T>
    __device__ T devide(T x) { return 1/x; }
    
    template<typename T>
    __device__ T Log(T x) { return log(x); }

    // ========== 함수 선언들 (구현은 d_matrix_2.cu에 있음) ==========
    
    // Zero padding
    template<typename T>
    __global__ void zeroPad(const T *d_A, T *d_C, int row, int col, int c_row, int c_col);

    template <typename T>
    d_matrix_2<T> zeroPedding(const d_matrix_2<T> &d_A, int size, cudaStream_t str = 0);
    
    // Hadamard Product
    template<typename T>
    __global__ void HPinKernel_1dx(const T* __restrict__ d_A, const T* __restrict__ d_B, T* __restrict__ C, int row, int col);
    
    template<typename T>
    __global__ void HPinKernel(const T* __restrict__ d_A, const T* __restrict__ d_B, T* __restrict__ d_C, int rows, int cols);
    
    template<typename T>
    d_matrix_2<T> HadamardProduct(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, cudaStream_t str=0);
    
    // Scalar Product
    template<typename T>
    __global__ void ScalainKernel(T* d_A, T scalar, T* d_C, int row, int col);
    
    template<typename T>
    d_matrix_2<T> ScalaProduct(const d_matrix_2<T>& d_A, T scalar, cudaStream_t str = 0);
    
    // Matrix Multiplication
    template<typename T>
    __global__ void MPinKernel(T* d_A, T* d_B, T* d_C, int row, int col, int eq);
    
    template<typename T>
    d_matrix_2<T> matrixMP(const d_matrix_2<T>& A, const d_matrix_2<T>& B, cudaStream_t str = 0);
    
    // Matrix Addition
    template<typename T>
    __global__ void PlusinKernel(T* d_A, T* d_B, T* d_C, int row, int col);
    
    template<typename T>
    d_matrix_2<T> matrixPlus(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, cudaStream_t str = 0);
    
    // Activation Functions
    template<typename T, T (*ActivateFunc)(T)>
    __global__ void ActivateInKernel(T* d_A, T* d_C, int row, int col);
     
    template<typename T, T (*ActivateFunc)(T)>
    d_matrix_2<T> MatrixActivate(const d_matrix_2<T>& d_A, cudaStream_t str = 0);
    
    // Scalar operations
    template<typename T>
    T plusAllElements(const d_matrix_2<T>& d_A);
    
    template<typename T>
    __global__ void plusScalaToMatrix(T* d_C,int row, int col, T scala);
    
    template<typename T>
    d_matrix_2<T> ScalaPlus(const d_matrix_2<T>& d_A, T scala);
    
    // Type casting
    template <typename T>
    __global__ void castKernel(const T* src, double* dst, int N);
    
    template <typename T>
    d_matrix_2<double> castToDoubleGPU(const d_matrix_2<T>& input);

    // Softmax functions
    template<typename T>
    __global__ void softmaxKernel(T* in, T* out, int row, int col);

    template<typename T>
    d_matrix_2<T> softmax(const d_matrix_2<T>& input, cudaStream_t str = 0);

    template<typename T>
    __global__ void efficientSoftmaxKernel(const T* in, T* out, int rows, int cols);
    
    template<typename T>
    d_matrix_2<T> softmax_efficient(const d_matrix_2<T>& input, cudaStream_t str = 0);

    // Utility functions
    template<typename T>
    __global__ void convertInKernel(T* d_X, int row, int col);
    
    template<typename T>
    d_matrix_2<T> convertZeroToEpsilon(d_matrix_2<T> x, cudaStream_t str = 0);

    // Convolution
    template<typename T>
    __global__ void convoluteInKernel(T* __restrict__ d_A, T* __restrict__ d_B, T* __restrict__ d_C, int inputRow, int inputCol, int filterRow, int filterCol, int outputRow, int outputCol, int stride);
    
    template<typename T>
    d_matrix_2<T> convolute(const d_matrix_2<T>& d_A, const d_matrix_2<T>& d_B, int stride, cudaStream_t str = 0);

    // Weight initialization
    __global__ void initCurandStates(curandState *states, unsigned long long seed, int total);
    
    template<typename T>
    __global__ void InitWeightInKernel(T* d_weight, curandState* states, int row, int col, InitType type);
    
    template<typename T>
    d_matrix_2<T> InitWeight(int row, int col, InitType type, cudaStream_t str = 0);

    // Matrix concatenation
    template<typename T>
    d_matrix_2<T> concatenate(const d_matrix_2<T>& A, const d_matrix_2<T>& B, cudaStream_t str = 0);

}//namespace d_matrix_ver2

#include <functional>
namespace std {
    template<typename T>
    struct hash<d_matrix_ver2::d_matrix_2<T>> {
        size_t operator()(d_matrix_ver2::d_matrix_2<T> const& m) const noexcept {
            size_t h = m.getRow() ^ (m.getCol() << 1);
            for (auto& val : m) {
                h ^= std::hash<T>{}(val) + 0x9e3779b97f4a7c15 + (h<<6) + (h>>2);
            }
            return h;
        }
    };
}
