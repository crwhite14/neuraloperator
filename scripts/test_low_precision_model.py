
import torch
import torch.nn as nn
import time
from tensorly import tenalg
tenalg.set_backend('einsum')
from pathlib import Path

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop import get_model, H1Loss

# Read the configuration
config_name = 'default'
#config_path = Path(__file__).parent.as_posix()
pipe = ConfigPipeline([YamlConfig('./factorized_config_renbo.yaml', config_name='default', config_folder='../config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='../config')
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

batch_size = config.data.batch_size
size = config.data.train_resolution

if torch.has_cuda:
    device = 'cuda'
else:
    device = 'cpu'

model = get_model(config)
model = model.to(device)
model.train()

in_data = torch.randn(batch_size, 3, size, size).to(device) #is this the right size for navier stokes?
in_label = torch.randn(batch_size, 1, size, size).to(device)
print(model.__class__)
#print(model)

t1 = time.time()
out = model(in_data)
t = time.time() - t1
print(f'Output of size {out.shape} in {t}.')

criterion = H1Loss(d=2)
loss = criterion(out.float(), in_label)
loss.backward()

'''
#simple profiling with command line printout
with torch.profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
    out = model(in_data)

print(prof.key_averages(group_by_stack_n=20).table(sort_by="cuda_time_total", row_limit=10))
'''


#generates profiling data in tensorboard

worker_name = 'mixed_precision_rank005_fwd'
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=10, warmup=100, active=100, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./factorized_log', worker_name=worker_name),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(1000):
        if step >= (10+100+120):
            break
        out = model(in_data)
        loss = criterion(out.float(), in_label)
        del loss
        del out
        #loss.backward()
        prof.step()

