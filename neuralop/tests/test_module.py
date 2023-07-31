import torch
import torch.nn as nn
import gc

# Let's define the Conv2D model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

# Function to calculate current GPU usage
def get_gpu_memory_usage(device):
    return torch.cuda.memory_allocated(device) / 1e6 # convert bytes to megabytes

# Function to test model with specified precision
def test_model_precision(dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # Clear GPU memory
    gc.collect() # Clear system memory
    model = ConvNet().to(device).type(dtype) # Move model to GPU and set precision
    input_tensor = torch.randn((64, 3, 128, 128)).to(device).type(dtype) # Move input tensor to GPU and set precision
    initial_memory = get_gpu_memory_usage(device)
    output = model(input_tensor)
    final_memory = get_gpu_memory_usage(device)
    del model, input_tensor, output # Delete references for clean-up
    torch.cuda.empty_cache() # Clear GPU memory
    gc.collect() # Clear system memory
    return final_memory - initial_memory

if __name__ == '__main__':
    print('Testing model with float32 precision...')
    memory_diff_32 = test_model_precision(torch.float32)
    print(f'Increase in GPU memory usage with float32 weights: {memory_diff_32}MB')
    
    print('Testing model with float16 precision...')
    memory_diff_16 = test_model_precision(torch.float16)
    print(f'Increase in GPU memory usage with float16 weights: {memory_diff_16}MB')
