#!/bin/bash

python train_navier_stokes.py --wandb.name=NS-10k-amp+half-tanh-tensorized --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --tfno2d.half_prec_inverse=True --opt.amp_autocast=True
