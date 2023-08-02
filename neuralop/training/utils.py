import subprocess
import time
import torch

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count

# Assuming using GPU 0
# device = torch.device('cuda:0')

def get_gpu_usage(device):
    #requires pynvml for this to work
    return torch.cuda.memory_allocated(device) / 1024**3 , torch.cuda.max_memory_allocated(device) / 1024**3 , torch.cuda.utilization(device)

def get_gpu_total_mem(device):
    return torch.cuda.get_device_properties(device).total_memory / 1024**3

def get_gpu_memory_and_utilization():
    # this hangs during the training loop because of "subprocess.check_output"
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Parse output
    gpu_memory_used, gpu_memory_total, gpu_utilization = result.strip().split(',')
    #gpu_memory_used_pct = float(gpu_memory_used) / float(gpu_memory_total)
    return int(gpu_memory_used), int(gpu_memory_total), int(gpu_utilization)

if __name__ == '__main__':
    #test utilization acquisition
    while True:
        #gpu_memory_used, gpu_memory_total, gpu_utilization = get_gpu_memory_and_utilization()
        #print('GPU Memory Used: {}/{} MiB'.format(gpu_memory_used, gpu_memory_total), 'GPU Utilization: {}%'.format(gpu_utilization))

        gpu_memory_allocated, gpu_memory_reserved, gpu_utilization = get_gpu_usage()
        gpu_total_mem = get_gpu_total_mem()
        print('GPU Total Memory: {} GB'.format(gpu_total_mem))
        print('GPU Memory Allocated/Reserved: {}/{} GB'.format(gpu_memory_allocated, gpu_memory_reserved), 'GPU Utilization: {}%'.format(gpu_utilization))
        time.sleep(1) 