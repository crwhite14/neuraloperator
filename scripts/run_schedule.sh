#!/bin/bash


#### now try the schedule (schedule is in config at [150-350])

python train_navier_stokes.py --wandb.name=NS-cp05-125-375 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --opt.n_epochs=700

python train_navier_stokes.py --wandb.name=NS-cp05-125-375 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=667 --opt.n_epochs=700

python train_navier_stokes.py --wandb.name=NS-cp05-125-375 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=668 --opt.n_epochs=700

##### rank 05, amp+full-fft-half

#python train_navier_stokes.py --wandb.name=NS-cp05-fullfft --tfno2d.stabilizer='full_fft' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

python train_navier_stokes.py --wandb.name=NS-cp05-fullfft --tfno2d.stabilizer='full_fft' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=667

python train_navier_stokes.py --wandb.name=NS-cp05-fullfft --tfno2d.stabilizer='full_fft' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=668

# test
python train_navier_stokes.py --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --data.n_train=1000
python train_navier_stokes.py --data.n_train=1000
# next:
python train_navier_stokes.py --data.n_train=1000 --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --opt.precision_schedule=[2,4]

# regular
#python train_navier_stokes.py --wandb.name=NS-cp512-regular --tfno2d.rank=512

#python train_navier_stokes.py --wandb.name=NS-cp256-regular --tfno2d.rank=256

#python train_navier_stokes.py --wandb.name=NS-cp128-regular --tfno2d.rank=128


