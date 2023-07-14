#!/bin/bash

# This script is used to retrain the neural operator for the Navier-Stokes
#python -m train_navier_stokes --tfno2d.rank=0.04 --tfno2d.norm='layer_norm' --wandb.name='smac_retrain_navier_stokes'

#python -m train_navier_stokes --tfno2d.rank=0.03 --tfno2d.factorization='cp' --tfno2d.n_modes_height=96 --tfno2d.n_modes_width=96 --data.batch_size=16  --wandb.name='smac_retrain_navier_stokes_2'

python -m train_navier_stokes --wandb.name='nas_conv_norm_retrain_navier_stokes'