#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <csignal>

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

    // Detach and remove the shared memory segment
    if (shared_flag) {
        shmdt(const_cast<int*>(shared_flag));
    }
    if (shmid != -1) {
        shmctl(shmid, IPC_RMID, NULL);
    }

    // Shutdown NVML
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

    // --- FIX IS HERE: New Threshold in Watts ---
    // This is the main tuning parameter. The controller will activate the
    // secondary workload if the GPU's power draw drops below this value.
    const unsigned int POWER_THRESHOLD_WATTS = 400; // 400 Watts

    const unsigned int POLLING_INTERVAL_US = 50000;

    // --- Setup Signal Handler for graceful exit ---
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // --- Setup Shared Memory ---
    key_t key = ftok("firefly_ipc_key", gpu_id);
    if (key == -1) {
        perror("ftok");
        return 1;
    }

    shmid = shmget(key, sizeof(int), IPC_CREAT | 0666);
    if (shmid == -1) {
        perror("shmget");
        return 1;
    }

    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) {
        perror("shmat");
        return 1;
    }
    
    *shared_flag = 0;
    std::cout << "Controller for physical GPU " << gpu_id << " started. Shared memory key: " << key << std::endl;

    // --- Setup NVML for GPU Monitoring ---
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device); 
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get handle for GPU " << gpu_id << " (logical index 0): " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    
    // --- Main Control Loop ---
    std::cout << "Monitoring power on physical GPU " << gpu_id << ". Press Ctrl+C to exit." << std::endl;
    while (true) {

        unsigned int power_milliwatts;
        result = nvmlDeviceGetPowerUsage(device, &power_milliwatts);

        if (result == NVML_SUCCESS) {
            // Convert milliwatts to watts for the comparison
            unsigned int power_watts = power_milliwatts / 1000;

            if (power_watts < POWER_THRESHOLD_WATTS) {
                *shared_flag = 1; // Signal RUN
            } else {
                *shared_flag = 0; // Signal STOP
            }
        } else {
            std::cerr << "Failed to get power usage for GPU " << gpu_id << ": " << nvmlErrorString(result) << std::endl;
            *shared_flag = 0; 
        }

        // Wait for the next polling interval
        usleep(POLLING_INTERVAL_US);
    }

    signalHandler(0);
    return 0;
}

