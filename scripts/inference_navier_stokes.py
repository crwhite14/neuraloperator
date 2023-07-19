import torch
import wandb
import sys
import copy
import os
import time
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop import get_model
from neuralop import Trainer
from neuralop.training import setup
from neuralop.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.utils import get_wandb_api_key, count_params, get_project_root, set_seed
from neuralop import LpLoss, H1Loss
from neuralop.models.spectral_convolution import FactorizedSpectralConv

from torch.nn.parallel import DistributedDataParallel as DDP

#from torch.ao.quantization import QConfigMapping
#from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
#from torch.ao.quantization.fx.custom_config import PrepareCustomConfig

# Note that this is temporary, we'll expose these functions to torch.ao.quantization after official releasee
#from torch.quantization.quantize_fx import prepare_fx, convert_fx

# ignore complexhalf warnings
import warnings
warnings.filterwarnings("ignore")

def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            #new = new.from_float(module)
            setattr(model, n, new)


# Read the configuration
config_name = 'default'
#config_folder = os.path.join(get_project_root(), 'config')
config_folder = os.path.join('..', 'config')
config_file_name = 'load_8layer_config.yaml'

pipe = ConfigPipeline([YamlConfig(config_file_name, config_name=config_name, config_folder=config_folder),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder=config_folder)
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set seed
if 'seed' in config and config.seed:
    print('setting seed to', config.seed)
    set_seed(config.seed)

#Set-up distributed communication, if using
device, is_logger = setup(config)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, output_encoder = load_navier_stokes_pt(
        config.data.folder, train_resolution=config.data.train_resolution, n_train=config.data.n_train, batch_size=config.data.batch_size, 
        positional_encoding=config.data.positional_encoding,
        test_resolutions=config.data.test_resolutions, n_tests=config.data.n_tests, test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input, encode_output=config.data.encode_output,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory, persistent_workers=config.data.persistent_workers
        )
model = get_model(config)
model = model.to(device)


#Use distributed data parallel 
if config.distributed.use_distributed:
    model = DDP(model,
                device_ids=[device.index],
                output_device=device.index,
                static_graph=True)


#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.opt.learning_rate, 
                                weight_decay=config.opt.weight_decay)

if config.opt.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
elif config.opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
elif config.opt.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.opt.step_size,
                                                gamma=config.opt.gamma)
else:
    raise ValueError(f'Got {config.opt.scheduler=}')

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == 'l2':
    train_loss = l2loss
elif config.opt.training_loss == 'h1':
    train_loss = h1loss
else:
    raise ValueError(f'Got training_loss={config.opt.training_loss} but expected one of ["l2", "h1"]')
eval_losses={'h1': h1loss, 'l2': l2loss}

trainer = Trainer(model, n_epochs=config.opt.n_epochs,
                  device=device,
                  mg_patching_levels=config.patching.levels,
                  mg_patching_padding=config.patching.padding,
                  mg_patching_stitching=config.patching.stitching,
                  wandb_log=config.wandb.log,
                  amp_autocast=config.opt.amp_autocast,
                  precision_schedule=config.opt.precision_schedule,
                  log_test_interval=config.wandb.log_test_interval,
                  log_output=config.wandb.log_output,
                  use_distributed=config.distributed.use_distributed,
                  verbose=config.verbose and is_logger)


# load model from dict
model_load_epoch = -1
file_path = 'checkpoint_best_2023-07-18.pt'
load_path = 'saved_checkpoints/' + file_path
trainer.load_model_checkpoint(model_load_epoch, model, optimizer, load_path=load_path)

#GPU warm-up
print('GPU warm-up')
trainer.evaluate(model, eval_losses, train_loader, output_encoder)

values_to_log = dict()

#time regular model (half-prec model)
for _ in range(3):
    msg = f'[{model_load_epoch}]'
    print('Time regular model (half-prec model)')
    start_inference = time.time()
    for loader_name, loader in test_loaders.items():

        errors = trainer.evaluate(model, eval_losses, loader, output_encoder, log_prefix=loader_name)

        for loss_name, loss_value in errors.items():
            msg += f', {loss_name}={loss_value:.4f}'
            values_to_log[loss_name] = loss_value

    end_inference = time.time()
    inference_time = end_inference - start_inference
    msg += f', inference_time={inference_time:.4f}'
    print(msg)
    print('Size of model before quantization (MB): ', get_size_of_model(model))
    print()

# todo: end here?

#time dynamic quantization model, eager mode
for _ in range(3):
    print('Time dynamic quantization model, eager mode')
    msg = f'[{model_load_epoch}]'
    model_to_optimize = copy.deepcopy(model)
    model_int8 = torch.ao.quantization.quantize_dynamic(model_to_optimize, {torch.nn.Linear}, dtype=torch.qint8)
    model_int8 = model_int8.to(device)
    start_inference = time.time()
    for loader_name, loader in test_loaders.items():

        errors = trainer.evaluate(model_int8, eval_losses, loader, output_encoder, log_prefix=loader_name)

        for loss_name, loss_value in errors.items():
            msg += f', {loss_name}={loss_value:.4f}'
            values_to_log[loss_name] = loss_value

    end_inference = time.time()
    inference_time = end_inference - start_inference
    msg += f', inference_time={inference_time:.4f}'
    print(msg)
    print('Size of model after quantization (MB): ', get_size_of_model(model_int8))
    print()


"""
#time dynamic quantization model, graph mode (fx)
print('Time dynamic quantization model, graph mode (fx)')
#get example input from test loader
example_inputs = next(iter(test_loaders[128]))

msg = f'[{model_load_epoch}]'
qconfig = get_default_qconfig_mapping("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)
prepare_custom_config_dict = {
    # option 1
    #"non_traceable_module_name": "FactorizedSpectralConv",
    # option 2
    "non_traceable_module_class": [FactorizedSpectralConv],
}

prepare_custom_config = PrepareCustomConfig()
prepare_custom_config.set_non_traceable_module_classes([FactorizedSpectralConv])

'''
this line is buggy because of implementation of qconfig_mapping
'''
prepared_model = prepare_fx(model, qconfig_mapping, example_inputs,
                            prepare_custom_config=prepare_custom_config)
# no calibration is required for dynamic quantization
model_int8_fx = convert_fx(prepared_model)  # convert the model to a dynamically quantized model
model_int8_fx = model_int8_fx.to(device)
start_inference = time.time()
for loader_name, loader in test_loaders.items():

    errors = trainer.evaluate(model_int8_fx, eval_losses, loader, output_encoder, log_prefix=loader_name)

    for loss_name, loss_value in errors.items():
        msg += f', {loss_name}={loss_value:.4f}'
        values_to_log[loss_name] = loss_value

end_inference = time.time()
inference_time = end_inference - start_inference
msg += f', inference_time={inference_time:.4f}'
print(msg)
print('Size of model after quantization (MB): ', get_size_of_model(model_int8_fx))
print()
"""

