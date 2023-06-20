#!/bin/bash


#regular FNO 
#python train_darcy.py --wandb.name=Darcy-5k-regular-main 

#both amp and half precision fourier without modifications
#python train_darcy.py --wandb.name=Darcy-5k-amp+half-main --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

#both amp and half precision fourier with tanh stabilizer 
#python train_darcy.py --wandb.name=Darcy-5k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True

for seed in 0 1 2
do 
    python train_darcy.py --wandb.name=Darcy-5k-regular-main-seed$seed --seed=$seed --distributed.seed=$seed

    python train_darcy.py --wandb.name=Darcy-5k-amp+half-tanh-main-seed$seed --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --seed=$seed --distributed.seed=$seed
done




