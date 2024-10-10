#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <cstring>  // For command-line argument parsing

// Global variables for configuration
bool use_multithreading = false;
bool use_simd = false;
bool use_cache_optimization = false;
bool use_all_three = false;

int num_threads = 1;  // Default to single thread
int block_size = 64;  // Default block size for cache blocking
int N = 1000;         // Matrix size (default to 1000x1000)
float density = 0.1;  // Matrix sparsity (default 10% non-zero elements)

std::mt19937 gen(42);  // Random number generator
std::uniform_real_distribution<float> dis(0.0, 1.0);

// Function to generate a sparse matrix
void generate_sparse_matrix(std::vector<std::vector<float>>& matrix, int size, float density) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = (dis(gen) < density) ? dis(gen) : 0.0f;
        }
    }
}

// Basic sequential matrix multiplication (no optimization)
void matrix_multiply(const std::vector<std::vector<float>>& A, 
                     const std::vector<std::vector<float>>& B, 
                     std::vector<std::vector<float>>& C, 
                     int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Multi-threaded matrix multiplication
void threaded_multiply(const std::vector<std::vector<float>>& A, 
                       const std::vector<std::vector<float>>& B, 
                       std::vector<std::vector<float>>& C) {
    int rows_per_thread = N / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? N : (t + 1) * rows_per_thread;

        threads.emplace_back(matrix_multiply, std::ref(A), std::ref(B), std::ref(C), start_row, end_row);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// Cache-optimized matrix multiplication (basic blocking)
void cache_optimized_multiply(const std::vector<std::vector<float>>& A, 
                              const std::vector<std::vector<float>>& B, 
                              std::vector<std::vector<float>>& C) {
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < std::min(i + block_size, N); ++ii) {
                    for (int jj = j; jj < std::min(j + block_size, N); ++jj) {
                        float sum = 0;
                        for (int kk = k; kk < std::min(k + block_size, N); ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

// SIMD-enhanced multiplication (partial example)
void simd_multiply(const std::vector<std::vector<float>>& A, 
                   const std::vector<std::vector<float>>& B, 
                   std::vector<std::vector<float>>& C, 
                   int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; j += 8) {  // Process 8 elements at once (AVX-256)
            __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector

            for (int k = 0; k < N; ++k) {
                __m256 a_vec = _mm256_broadcast_ss(&A[i][k]);  // Broadcast a[i][k] to all lanes
                __m256 b_vec = _mm256_loadu_ps(&B[k][j]);       // Load B[k][j:j+7]
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec); // Multiply and accumulate
            }

            // Store the result back into C
            _mm256_storeu_ps(&C[i][j], sum_vec);
        }
    }
}

// Dispatcher for optimization techniques
void multiply_matrices(const std::vector<std::vector<float>>& A, 
                       const std::vector<std::vector<float>>& B, 
                       std::vector<std::vector<float>>& C) {
    if (use_multithreading) {
        threaded_multiply(A, B, C);
    } else if (use_cache_optimization) {
        cache_optimized_multiply(A, B, C);
    } else if (use_simd) {
        simd_multiply(A, B, C, 0, N);
    } else {
        matrix_multiply(A, B, C, 0, N);
    }
}

// Function for multiplying matrix blocks using SIMD within a thread
void simd_multiply_block(const std::vector<std::vector<float>>& A,
                         const std::vector<std::vector<float>>& B,
                         std::vector<std::vector<float>>& C,
                         int start_row, int end_row, int n) {
    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // Ensure we don't go out of bounds
                for (int ii = i; ii < std::min(i + block_size, end_row); ++ii) {
                    for (int jj = j; jj < std::min(j + block_size, n); jj += 8) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        for (int kk = k; kk < std::min(k + block_size, n); ++kk) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[ii][kk]);
                            __m256 b_vec = _mm256_loadu_ps(&B[kk][jj]);
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        _mm256_storeu_ps(&C[ii][jj], sum_vec);
                    }
                }
            }
        }
    }
}

void parallel_multiply(const std::vector<std::vector<float>>& A,
                       const std::vector<std::vector<float>>& B,
                       std::vector<std::vector<float>>& C,
                       int num_threads) {
    int n = A.size(); // Assume square matrices
    std::vector<std::thread> threads;

    int rows_per_thread = n / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        threads.push_back(std::thread(simd_multiply_block, std::cref(A), std::cref(B), std::ref(C), start_row, end_row, n));
    }

    // Join threads
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}



int main(int argc, char* argv[]) {
    // Parse command-line arguments
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
                N = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--density") == 0 && i + 1 < argc) {
                density = std::stof(argv[++i]);
            } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
                use_multithreading = true;
            } else if (strcmp(argv[i], "--simd") == 0) {
                use_simd = true;
            } else if (strcmp(argv[i], "--cache") == 0) {
                use_cache_optimization = true;
            }else if (strcmp(argv[i], "--all") == 0) {
                use_all_three = true;
            }
        }
    }

    // Initialize matrices
    std::vector<std::vector<float>> A(N, std::vector<float>(N));
    std::vector<std::vector<float>> B(N, std::vector<float>(N));
    std::vector<std::vector<float>> C(N, std::vector<float>(N, 0.0f));

    // Generate sparse matrices
    generate_sparse_matrix(A, N, density);
    generate_sparse_matrix(B, N, density);

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    if (use_all_three == true){
        parallel_multiply(A, B, C, num_threads);
    } else{
        multiply_matrices(A, B, C);
    }
   
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Size: " << N << "x" << N <<std::endl;
    std::cout << "Density: " << density*100 << "%" <<std::endl;
    std::cout << "Threads: " << num_threads << "\n" << std::endl;
    std::cout << "Matrix multiplication completed in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
