#!/usr/bin/env bash

# gpu_logger.sh
# Logs GPU temperature, utilization, and power draw into a CSV file

OUT_FILE="gpu_metrics.csv"
INTERVAL_MS=100   # sampling interval in ms

# Write CSV header
echo "timestamp,gpu_index,temperature_C,utilization_gpu_percent,power_W" > "$OUT_FILE"

# Loop forever until Ctrl+C
while true; do
  # Current timestamp in ISO8601
  ts=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

  # Query all GPUs
  nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw \
             --format=csv,noheader,nounits \
             | while IFS=, read -r idx temp util power; do
                 echo "$ts,$idx,$temp,$util,$power" >> "$OUT_FILE"
               done

  # Sleep interval (convert ms to s)
  sleep 0.$((INTERVAL_MS/10))
done
