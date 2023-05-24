#!/bin/bash


#regular FNO 
python train_darcy.py --wandb.name=Darcy-5k-regular-main 

#both amp and half precision fourier with tanh stabilizer 
python train_darcy.py --wandb.name=Darcy-5k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True


#regular FNO 
python train_darcy.py --wandb.name=Darcy-5k-regular-main  --distributed.seed=667

#both amp and half precision fourier with tanh stabilizer 
python train_darcy.py --wandb.name=Darcy-5k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=667


#regular FNO 
python train_darcy.py --wandb.name=Darcy-5k-regular-main --distributed.seed=668

#both amp and half precision fourier with tanh stabilizer 
python train_darcy.py --wandb.name=Darcy-5k-amp+half-tanh-main --tfno2d.stabilizer='tanh' --tfno2d.half_prec_fourier=True --opt.amp_autocast=True --distributed.seed=668


