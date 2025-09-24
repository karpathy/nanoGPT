#include <iostream>
#include <stdexcept>
#include <csignal>

// For system calls
#include <unistd.h>

// For shared memory (IPC)
#include <sys/ipc.h>
#include <sys/shm.h>

// For CUDA and cuBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper to check for CUDA errors
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Global pointers for cleanup
volatile int* shared_flag = nullptr;
float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
cublasHandle_t cublas_handle;

// Signal handler for clean shutdown
void signalHandler(int signum) {
    std::cout << "\nSecondary workload shutting down..." << std::endl;
    
    // Detach shared memory
    if (shared_flag) {
        shmdt(const_cast<int*>(shared_flag));
    }

    // Free GPU memory
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    // Destroy cuBLAS handle
    if (cublas_handle) cublasDestroy(cublas_handle);

    exit(signum);
}

int main(int argc, char* argv[]) {
    // --- Configuration ---
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --gpu-id <id>" << std::endl;
        return 1;
    }
    int gpu_id = std::stoi(argv[2]);

    const int M = 8192;
    const int N = 8192;
    const int K = 8192;

    // --- Setup Signal Handler ---
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // --- Setup Shared Memory ---
    key_t key = ftok("firefly_ipc_key", gpu_id);
    if (key == -1) {
        perror("ftok");
        return 1;
    }

    int shmid = shmget(key, sizeof(int), 0666);
    if (shmid == -1) {
        perror("shmget");
        return 1;
    }

    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) {
        perror("shmat");
        return 1;
    }
    std::cout << "Secondary workload for physical GPU " << gpu_id << " attached to shared memory." << std::endl;

    // --- Setup CUDA and cuBLAS ---
    // --- FIX IS HERE ---
    // When CUDA_VISIBLE_DEVICES is set to a single GPU, that GPU always appears
    // as device index 0 to the program.
    CUDA_CHECK(cudaSetDevice(0));

    // Allocate matrices on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeof(float) * M * N));

    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
    
    // GEMM calculation constants
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --- Main Workload Loop ---
    std::cout << "Waiting for signal from controller..." << std::endl;
    while(true) {
        if (*shared_flag == 1) {
            cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        d_A, M,
                        d_B, K,
                        &beta,
                        d_C, M);
        } else {
            usleep(1000); // Sleep for 1ms
        }
    }

    signalHandler(0);
    return 0;
}
