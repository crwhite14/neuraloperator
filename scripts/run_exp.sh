#!/bin/bash

#tensorized with MP 
#python train_navier_stokes.py --wandb.name=NS-10k-mp-tensorized-1024test --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --tfno2d.half_prec_inverse=True --opt.amp_autocast=True

#tensorized with FP 
#python train_navier_stokes.py --wandb.name=NS-10k-full-tensorized-1024test

#tensorized with FP + changed preactivation
python train_navier_stokes.py --tfno2d.half_prec_fourier=True --tfno2d.half_prec_inverse=True --opt.amp_autocast=True --wandb.name=NS-10k-full-tensorized-preact-twoAct
