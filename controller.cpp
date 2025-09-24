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
        shmdt(shared_flag);
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

    // This is the main tuning parameter. The controller will activate the
    // secondary workload if GPU utilization drops below this percentage.
    const unsigned int UTILIZATION_THRESHOLD = 50; // 50%

    // Polling rate. 50,000 microseconds = 50ms = 20Hz.
    const unsigned int POLLING_INTERVAL_US = 50000;

    // --- Setup Signal Handler for graceful exit ---
    signal(SIGINT, signalHandler);  // Catches Ctrl+C
    signal(SIGTERM, signalHandler); // Catches kill command

    // --- Setup Shared Memory ---
    // We use the GPU ID to create a unique key for the shared memory segment.
    // This ensures each controller/worker pair has its own private channel.
    key_t key = ftok("firefly_ipc_key", gpu_id);
    if (key == -1) {
        perror("ftok");
        return 1;
    }

    // Get a shared memory segment of the size of an integer.
    // IPC_CREAT | 0666 creates the segment if it doesn't exist and sets permissions.
    shmid = shmget(key, sizeof(int), IPC_CREAT | 0666);
    if (shmid == -1) {
        perror("shmget");
        return 1;
    }

    // Attach the segment to our process's address space.
    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) {
        perror("shmat");
        return 1;
    }
    
    // Initialize the flag to 0 (STOP)
    *shared_flag = 0;
    std::cout << "Controller for GPU " << gpu_id << " started. Shared memory key: " << key << std::endl;

    // --- Setup NVML for GPU Monitoring ---
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get handle for GPU " << gpu_id << ": " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    
    // --- Main Control Loop ---
    std::cout << "Monitoring GPU " << gpu_id << ". Press Ctrl+C to exit." << std::endl;
    while (true) {
        nvmlUtilization_t util;
        result = nvmlDeviceGetUtilizationRates(device, &util);

        if (result == NVML_SUCCESS) {
            // Decision logic: if GPU compute is below the threshold, set flag to 1 (RUN).
            // Otherwise, set it to 0 (STOP).
            if (util.gpu < UTILIZATION_THRESHOLD) {
                *shared_flag = 1; // Signal RUN
            } else {
                *shared_flag = 0; // Signal STOP
            }
        } else {
            std::cerr << "Failed to get utilization for GPU " << gpu_id << ": " << nvmlErrorString(result) << std::endl;
            // In case of error, default to STOP for safety
            *shared_flag = 0; 
        }

        // Wait for the next polling interval
        usleep(POLLING_INTERVAL_US);
    }

    // The signal handler will take care of cleanup, but this is good practice
    signalHandler(0);
    return 0;
}
