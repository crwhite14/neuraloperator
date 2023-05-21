
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
pipe = ConfigPipeline([YamlConfig('./best_config_full.yaml', config_name='default', config_folder='../config'),
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

worker_name = 'half_precision_torch2'

'''
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=10, warmup=100, active=50, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./halfprec_log', worker_name=worker_name),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(1000):
        if step >= (10+100+626):
            break
        out = model(in_data)
        loss = criterion(out.float(), in_label)
        loss.backward()
        prof.step()

'''