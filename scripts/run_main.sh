#!/bin/bash

# Run ablations experiments for the paper

#regular FNO 
#python train_navier_stokes.py --wandb.name=NS-10k-regular-main 

#autocast only
#python train_navier_stokes.py --wandb.name=NS-10k-amp-only-main --opt.amp_autocast=True

#both amp and half precision fourier with full-precision rfft
python train_navier_stokes.py --wandb.name=NS-10k-amp+half-full-rfft-main --tfno2d.stabilizer='full_fft'  --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

