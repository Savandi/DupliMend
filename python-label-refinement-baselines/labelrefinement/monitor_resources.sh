#!/bin/bash

# Resource monitoring script
# Run this in a separate terminal while processing runs

echo "Resource Monitor - Press Ctrl+C to stop"
echo "=========================================="
echo ""

LOG_FILE="./monitoring_logs/resources_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./monitoring_logs

echo "Logging to: $LOG_FILE"
echo ""

# Function to get memory usage
get_memory() {
    free -h | grep Mem | awk '{print $3 "/" $2}'
}

# Function to get temp disk usage
get_temp_disk() {
    du -sh /tmp 2>/dev/null | awk '{print $1}' || echo "N/A"
}

# Function to get process count
get_python_procs() {
    pgrep -c python 2>/dev/null || echo "0"
}

# Header
printf "%-20s | %-15s | %-15s | %-10s\n" "Timestamp" "Memory Used" "/tmp Size" "Python Procs" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"

# Monitor loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    MEMORY=$(get_memory)
    TEMP=$(get_temp_disk)
    PROCS=$(get_python_procs)

    printf "%-20s | %-15s | %-15s | %-10s\n" "$TIMESTAMP" "$MEMORY" "$TEMP" "$PROCS" | tee -a "$LOG_FILE"

    sleep 10
done
