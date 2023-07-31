
import torch
import time
from tensorly import tenalg
tenalg.set_backend('einsum')
from pathlib import Path
from torch.cuda import amp

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop import get_model

# Read the configuration
config_name = 'default'
config_path = Path(__file__).parent.as_posix()
pipe = ConfigPipeline([YamlConfig('./test_config.yaml', config_name='default', config_folder=config_path),
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

batch_size = config.data.batch_size
size = config.data.size

if torch.has_cuda:
    device = 'cuda'
else:
    device = 'cpu'

model = get_model(config)
model = model.to(device)

in_data = torch.randn(batch_size, 3, size, size).to(device)
print(model.__class__)
#print(model)

t1 = time.time()
start_mem = torch.cuda.memory_allocated()/1e9

if config.opt.amp_autocast:
    print('Using autocast.')
    with amp.autocast(enabled=True):
        out = model(in_data)
else:
    out = model(in_data)
end_mem = torch.cuda.memory_allocated()/1e9
print(f'Memory usage: {end_mem-start_mem} GB')
t = time.time() - t1
print(f'Output of size {out.shape} in {t}.')

loss = out.sum()
loss.backward()
