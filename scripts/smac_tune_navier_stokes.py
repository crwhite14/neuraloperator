import torch
import wandb
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from ConfigSpace import ConfigurationSpace
from ConfigSpace import (
    Categorical,
    Configuration,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.acquisition.function import EI, LCB

from neuralop import get_model
from neuralop import TunedTrainer
from neuralop.training import setup
from neuralop.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

# Read the configuration
config_name = 'default'
config_folder = os.path.join('..', 'config')
config_file_name = 'tuning_navier_stokes_config.yaml'

pipe = ConfigPipeline([YamlConfig(config_file_name, config_name=config_name, config_folder=config_folder),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder=config_folder)
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

#Set-up distributed communication, if using
device, is_logger = setup(config)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger
#Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, val_loader, output_encoder = load_navier_stokes_pt(
        config.data.folder, train_resolution=config.data.train_resolution, n_train=config.data.n_train, batch_size=config.data.batch_size, 
        positional_encoding=config.data.positional_encoding, val_split=config.data.val_split, 
        test_resolutions=config.data.test_resolutions, n_tests=config.data.n_tests, test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input, encode_output=config.data.encode_output,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory, persistent_workers=config.data.persistent_workers
        )

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

# plot smac optimization trajectory
def plot_trajectory(facades: list[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.4)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()
    plt.savefig('smac_trajectory.png')

#setup training method
def train_model(tune_cfg, seed=0, budget=10):
    config = pipe.read_conf()
    tune_dict = tune_cfg.get_dictionary()
    config.tfno2d.rank= tune_dict['tensorization_rank']
    config.tfno2d.factorization = tune_dict['factorization_method']
    config.tfno2d.n_modes_height = tune_dict['frequency_mode_factor'] * 16
    config.tfno2d.n_modes_width = tune_dict['frequency_mode_factor'] * 16
    config.data.batch_size = 2 ** (tune_dict['log_batch_size'])
    print(tune_dict)

    model = get_model(config)
    model = model.to(device)

    #Log parameter count
    if is_logger:
        n_params = count_params(model)

        if config.verbose:
            print(f'\nn_params: {n_params}')
            sys.stdout.flush()

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

    trainer = TunedTrainer(model, n_epochs=int(budget),
                    device=device,
                    mg_patching_levels=config.patching.levels,
                    mg_patching_padding=config.patching.padding,
                    mg_patching_stitching=config.patching.stitching,
                    wandb_log=config.wandb.log,
                    amp_autocast=config.opt.amp_autocast,
                    log_test_interval=config.wandb.log_test_interval,
                    log_output=config.wandb.log_output,
                    use_distributed=config.distributed.use_distributed,
                    verbose=config.verbose)


    last_five_errors = trainer.train(train_loader, val_loader,
                output_encoder,
                model, 
                optimizer,
                scheduler, 
                regularizer=False, 
                training_loss=train_loss,
                eval_losses=eval_losses)

    sum_loss = 0
    for errors in last_five_errors:
        sum_loss += list(errors.values())[0]
        for key in errors.keys():
            print(f'{key}: {errors[key]}')
    
    return sum_loss / 5.0

def configspace() -> ConfigurationSpace:
     
    cs = ConfigurationSpace()
    tensorization_rank = Float('tensorization_rank', (0.01, 0.1), default=0.05)
    factorization_method = Categorical('factorization_method', ['cp', 'tucker', 'tt', 'dense'], default='cp')
    frequency_mode_factor = Integer('frequency_mode_factor', (1, 6), default=4)
    log_batch_size = Integer('log_batch_size', (4, 6), default=6)

    cs.add_hyperparameters([tensorization_rank, factorization_method, frequency_mode_factor, log_batch_size])

    return cs 

if __name__ == '__main__':
    facades: list[AbstractFacade] = []
    for intensifier_object in [Hyperband]:
        #budget setting following table1 in https://openreview.net/pdf?id=ry18Ww5ee
        scenario_name = f'halfprec_tensorization_rank_mode_bs_search_{intensifier_object.__name__}'
        scenario = Scenario(configspace(),
                            name=scenario_name,
                            n_trials=100, 
                            min_budget=9, 
                            max_budget=81)

        initial_design = MFFacade.get_initial_design(scenario, n_configs=2)

        intensifier = intensifier_object(scenario=scenario)
        acquisition_function = EI()

        smac = MFFacade(
            scenario,
            train_model,
            initial_design=initial_design,
            intensifier=intensifier,
            acquisition_function=acquisition_function,
            overwrite=True,
        )

        incumbent = smac.optimize()

        best_tensorization_rank = incumbent.get('tensorization_rank')
        best_factorization_method = incumbent.get('factorization_method')
        best_freq_mode = incumbent.get('frequency_mode_factor')
        best_batch_size = incumbent.get('log_batch_size')
        
        print('Intensifier is ', intensifier_object)
        print(f'Best tensorization_rank: {best_tensorization_rank}')
        print(f'Best factorization_method: {best_factorization_method}')
        print(f'Best factor of freqency modes: {best_freq_mode}')
        print(f'Best log batch size: {best_batch_size}')

        with open(f'./smac3_output/{scenario_name}_best_config.txt', 'w') as f:
            f.write(f'Best tensorization_rank: {best_tensorization_rank}\n')
            f.write(f'Best factorization_method: {best_factorization_method}\n')
            f.write(f'Best factor (16) of freqency modes: {best_freq_mode}\n')
            f.write(f'Best log batch size: {best_batch_size}\n')

        facades.append(smac)

    plot_trajectory(facades)




