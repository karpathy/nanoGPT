#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <csignal>
#include <fstream>  // For file I/O
#include <chrono>   // For timestamps

// For system calls and timing
#include <unistd.h>

// For shared memory (IPC)
#include <sys/ipc.h>
#include <sys/shm.h>

// For NVIDIA Management Library (NVML)
#include <nvml.h>

// Global pointer to the shared memory segment
volatile int* shared_flag = nullptr;
int shmid = -1; // Shared memory ID

// Signal handler to ensure clean shutdown
void signalHandler(int signum) {
    std::cout << "\nController shutting down..." << std::endl;
    if (shared_flag) {
        shmdt(const_cast<int*>(shared_flag));
    }
    if (shmid != -1) {
        shmctl(shmid, IPC_RMID, NULL);
    }
    nvmlShutdown();
    exit(signum);
}

int main(int argc, char* argv[]) {
    // --- Configuration ---
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --gpu-id <id>" << std::endl;
        return 1;
    }
    int gpu_id = std::stoi(argv[2]);

    // This is the main tuning parameter.
    const unsigned int POWER_THRESHOLD_WATTS = 250; 

    const unsigned int POLLING_INTERVAL_US = 50000;

    // --- Setup ---
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // --- NEW: Setup CSV Logging ---
    std::ofstream log_file;
    std::string log_filename = "flag_log_gpu_" + std::to_string(gpu_id) + ".csv";
    log_file.open(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not open log file " << log_filename << std::endl;
        return 1;
    }
    log_file << "timestamp_ms,flag_state\n";

    // --- Setup Shared Memory ---
    key_t key = ftok("firefly_ipc_key", gpu_id);
    if (key == -1) { perror("ftok"); return 1; }
    shmid = shmget(key, sizeof(int), IPC_CREAT | 0666);
    if (shmid == -1) { perror("shget"); return 1; }
    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) { perror("shmat"); return 1; }
    *shared_flag = 0;
    std::cout << "Controller for GPU " << gpu_id << " started with logging." << std::endl;

    // --- Setup NVML ---
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) { std::cerr << "NVML Init Error" << std::endl; return 1; }
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) { std::cerr << "NVML Handle Error" << std::endl; nvmlShutdown(); return 1; }
    
    // --- Main Control Loop ---
    std::cout << "Monitoring power on GPU " << gpu_id << ". Press Ctrl+C to exit." << std::endl;
    while (true) {
        unsigned int power_milliwatts;
        result = nvmlDeviceGetPowerUsage(device, &power_milliwatts);

        if (result == NVML_SUCCESS) {
            unsigned int power_watts = power_milliwatts / 1000;
            int new_flag_state = 0;

            if (power_watts < POWER_THRESHOLD_WATTS) {
                new_flag_state = 1; // Propose RUN
            } else {
                new_flag_state = 0; // Propose STOP
            }

            // --- MODIFIED: Log every polling period ---
            auto now = std::chrono::system_clock::now();
            auto ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            log_file << ms_since_epoch << "," << new_flag_state << "\n";
            
            *shared_flag = new_flag_state;

        } else {
            std::cerr << "Failed to get power for GPU " << gpu_id << ": " << nvmlErrorString(result) << std::endl;
            *shared_flag = 0; 
        }

        usleep(POLLING_INTERVAL_US);
    }

    signalHandler(0);
    return 0;
}

