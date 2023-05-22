#!/bin/bash

# Run ablations experiments for the paper

#regular FNO 
#python train_navier_stokes.py --wandb.name=NS-10k-regular 

#autocast only
#python train_navier_stokes.py --wandb.name=NS-10k-amp-only --opt.amp_autocast=True

#half precision fourier only
#python train_navier_stokes.py --wandb.name=NS-10k-half-only --tfno2d.half_prec_fourier=True

#both amp and half precision fourier without modifications
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with full-precision rfft
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-full-rfft --tfno2d.stabilizer='full_fft'  --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with hard clipping
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-hard-clip --tfno2d.stabilizer='clip_hard' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with 2-sigma clipping
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-2sigma-clip --tfno2d.stabilizer='clip_sigma' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with tanh
#python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with interpolation
#TBD

#full-precision with n_modes_height = 32
#python train_navier_stokes.py --wandb.name=NS-10k-regular-mode32 --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32

#full-precision with n_modes_height = 16 
#python train_navier_stokes.py --wandb.name=NS-10k-regular-mode16 --tfno2d.n_modes_height=16 --tfno2d.n_modes_width=16

#half-preciion with n_modes_height = 32
#python train_navier_stokes.py --wandb.name=NS-10k-half-mode32-tanh --tfno2d.n_modes_height=32 --tfno2d.n_modes_width=32 --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.stabilizer='tanh'

#half-preciion with n_modes_height = 16
#python train_navier_stokes.py --wandb.name=NS-10k-half-mode16-tanh --tfno2d.n_modes_height=16 --tfno2d.n_modes_width=16 --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.stabilizer='tanh'
