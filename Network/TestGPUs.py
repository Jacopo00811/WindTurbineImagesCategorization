# Imports
import torch
import os

# Show available devices
num_devices = torch.cuda.device_count()
print(f"Number of available CUDA-enabled GPUs: {num_devices}")

# Set CUDA_VISIBLE_DEVICES variable to control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_devices))

# Device selection based on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")
