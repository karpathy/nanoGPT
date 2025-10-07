#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <csignal>
#include <fstream>
#include <chrono>

// For system calls and timing
#include <unistd.h>

// For shared memory (IPC)
#include <sys/ipc.h>
#include <sys/shm.h>

// Global pointer to the shared memory segment
volatile int* shared_flag = nullptr;
int shmid = -1; // Shared memory ID

// Signal handler to ensure clean shutdown
void signalHandler(int signum) {
    std::cout << "\nController shutting down..." << std::endl;
    if (shared_flag) {
        // Ensure flag is set to 0 on exit
        *shared_flag = 0;
        shmdt(const_cast<int*>(shared_flag));
    }
    if (shmid != -1) {
        shmctl(shmid, IPC_RMID, NULL);
    }
    exit(signum);
}

int main(int argc, char* argv[]) {
    // --- Configuration ---
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --gpu-id <id>" << std::endl;
        return 1;
    }
    int gpu_id = std::stoi(argv[2]);

    // --- NEW: Periodic Configuration ---
    // The duration to keep the workload active.
    const unsigned int ACTIVATION_DURATION_US = 50000; // 50ms
    // The total period for the cycle.
    const unsigned int PERIOD_SECONDS = 300;
    // The remaining time to sleep to maintain the 10-second period.
    const unsigned int SLEEP_DURATION_US = (PERIOD_SECONDS * 1000000) - ACTIVATION_DURATION_US;

    // --- Setup ---
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // --- Setup CSV Logging ---
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
    if (shmid == -1) { perror("shmget"); return 1; }
    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) { perror("shmat"); return 1; }
    *shared_flag = 0;
    std::cout << "Periodic controller for GPU " << gpu_id << " started." << std::endl;

    // --- Main Control Loop ---
    std::cout << "Issuing one workload every " << PERIOD_SECONDS << " seconds. Press Ctrl+C to exit." << std::endl;
    while (true) {
        // --- Activate workload ---
        *shared_flag = 1;
        auto now_on = std::chrono::system_clock::now();
        auto ms_on = std::chrono::duration_cast<std::chrono::milliseconds>(now_on.time_since_epoch()).count();
        log_file << ms_on << ",1\n" << std::flush; // Flush to ensure it's written immediately

        usleep(ACTIVATION_DURATION_US);

        // --- Deactivate workload ---
        *shared_flag = 0;
        auto now_off = std::chrono::system_clock::now();
        auto ms_off = std::chrono::duration_cast<std::chrono::milliseconds>(now_off.time_since_epoch()).count();
        log_file << ms_off << ",0\n" << std::flush;

        // --- Sleep until next cycle ---
        usleep(SLEEP_DURATION_US);
    }

    signalHandler(0);
    return 0;
}

