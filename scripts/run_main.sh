#!/bin/bash

#regular FNO 
#python train_navier_stokes.py --wandb.name=NS-10k-regular-main 

#regular FNO + 32 modes
#python train_navier_stokes.py --wandb.name=NS-10k-regular-mode32-main --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32

#both amp and half precision fourier with tanh stabilizer 
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with tanh stabilizer + 32 modes
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-mode32-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32

#both amp and half precision fourier with clip stabilizer + 64 modes (we're not using this result)
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-clip-main --tfno2d.stabilizer='clip_hard' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True


#second random seed for the experiments above 
#regular FNO 
#python train_navier_stokes.py --wandb.name=NS-10k-regular-main-seed2 

#regular FNO + 32 modes
#python train_navier_stokes.py --wandb.name=NS-10k-regular-mode32-main-seed2 --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32

#both amp and half precision fourier with tanh stabilizer 
python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-main-seed2 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --tfno2d.half_prec_inverse=True --opt.amp_autocast=True

#both amp and half precision fourier with tanh stabilizer + 32 modes
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-mode32-main-seed2 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32