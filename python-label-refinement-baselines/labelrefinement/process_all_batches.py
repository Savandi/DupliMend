import subprocess
import os

log_size = 500
batch_size = 30
exp_nr = 1
total_datasets = 647
num_batches = (total_datasets + batch_size - 1) // batch_size  # Ceiling division

for batch_num in range(num_batches):
    print(f"Starting batch {batch_num + 1}/{num_batches}")
    subprocess.run([
        "python", "test_main.py", 
        str(log_size), str(batch_size), str(exp_nr), str(batch_num)
    ])
    print(f"Completed batch {batch_num + 1}")

print("All batches completed!")