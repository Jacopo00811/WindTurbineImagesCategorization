import torch
num_dev = torch.cuda.device_count()
print(f"Num of dev: {num_dev}")
