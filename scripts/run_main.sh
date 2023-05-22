#!/bin/bash

# Run ablations experiments for the paper

#regular FNO 
#python train_navier_stokes.py --wandb.name=NS-10k-regular-main 

#regular FNO + 32 modes
python train_navier_stokes.py --wandb.name=NS-10k-regular-mode32-main --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32

#both amp and half precision fourier with tanh stabilizer 
python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with tanh stabilizer + 32 modes
python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-mode32-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32
