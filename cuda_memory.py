import torch

# some code that uses CUDA memory

# get current memory usage
print(f"Current CUDA memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

# get maximum memory usage
print(f"Maximum CUDA memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
