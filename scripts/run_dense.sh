#!/bin/bash


python train_navier_stokes.py --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --tfno2d.stabilizer='tanh' --data.n_train=500 --opt.n_epochs=50 --tfno2d.factorization=None

