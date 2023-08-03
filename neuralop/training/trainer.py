import torch
from torch.cuda import amp
from torch.cuda.amp import GradScaler
from timeit import default_timer
import wandb
import sys
import os
import pandas as pd
from datetime import datetime
import pandas as pd

import neuralop.mpu.comm as comm

from .patching import MultigridPatching2D
from .losses import LpLoss
from .utils import AverageMeter, get_gpu_total_mem, get_gpu_usage, get_gpu_memory_map


class Trainer:
    def __init__(self, model, n_epochs, wandb_log=True, amp_autocast=False, grad_clip=False, device=None,
                 mg_patching_levels=0, mg_patching_padding=0, mg_patching_stitching=True, precision_schedule=None,
                 log_test_interval=1, log_output=False, use_distributed=False, save_interval=10, model_save_dir='./checkpoints', 
                 verbose=True):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        amp_autocast: bool, default is False
        grad_clip: float
            if 0/False, no gradient clipping is done
            if > 0, indicates the maximum allowed norm of the gradients
        precision_schedule: array of ints
            sets Fourier layer precision to half, mixed, full
        device : torch.device
        mg_patching_levels : int, default is 0
            if 0, no multi-grid domain decomposition is used
            if > 0, indicates the number of levels to use
        mg_patching_padding : float, default is 0
            value between 0 and 1, indicates the fraction of size to use as padding on each side
            e.g. for an image of size 64, padding=0.25 will use 16 pixels of padding on each side
        mg_patching_stitching : bool, default is True
            if False, the patches are not stitched back together and the loss is instead computed per patch
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        save_interval : int, default is 10
            how frequently to save checkpoints
        model_save_dir : str, default is './checkpoints'
        verbose : bool, default is True
        """
        self.n_epochs = n_epochs
        self.wandb_log = wandb_log
        self.amp_autocast = amp_autocast
        self.grad_clip = grad_clip
        # todo: currently setting an argument of array as ints is not working, neither in yaml file or as command line arg
        if precision_schedule:
            try:
                self.precision_schedule = [int(x) for x in list(precision_schedule)]
            except:
                self.precision_schedule = [int(x) for x in ''.join(precision_schedule)[1:-1].split(',')]
        else:
            self.precision_schedule = None

        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.mg_patching_levels = mg_patching_levels
        self.mg_patching_stitching = mg_patching_stitching
        self.use_distributed = use_distributed
        self.device = device
        self.save_interval = save_interval
        self.model_save_dir = model_save_dir
        #create model save dir if not exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        

        if mg_patching_levels > 0:
            self.mg_n_patches = 2**mg_patching_levels
            if verbose:
                print(f'Training on {self.mg_n_patches**2} multi-grid patches.')
                sys.stdout.flush()
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                print(f'Training on regular inputs (no multi-grid patching).')
                sys.stdout.flush()

        self.mg_patching_padding = mg_patching_padding
        self.patcher = MultigridPatching2D(model, levels=mg_patching_levels, padding_fraction=mg_patching_padding,
                                           use_distributed=use_distributed, stitching=mg_patching_stitching)

    def train(self, train_loader, test_loaders, output_encoder,
              model, optimizer, scheduler, regularizer, 
              training_loss=None, eval_losses=None):
        """Trains the given model on the given datasets"""
        n_train = len(train_loader.dataset)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if self.verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if output_encoder is not None:
            output_encoder.to(self.device)
        
        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 


        GPU_memory_meter_macro = AverageMeter()
        GPU_util_meter_macro = AverageMeter()
        gpu_mem_capacity = get_gpu_total_mem(self.device)
        time_meter = AverageMeter()
        measure_gpu = True
        scaler = GradScaler(enabled=self.amp_autocast) 
        for epoch in range(self.n_epochs):
            avg_loss = 0
            avg_lasso_loss = 0
            model.train()
            t1 = default_timer()
            train_err = 0.0
            GPU_memory_meter_micro = AverageMeter()
            GPU_util_meter_micro = AverageMeter()
            measure_gpu = True

            for idx, sample in enumerate(train_loader):
                x, y = sample['x'], sample['y']

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')

                x, y = self.patcher.patch(x, y)

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. patched inputs of size {x.shape=}, {y.shape=}')

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = model(x)
                else:
                    out = model(x)

                #first measurement
                if measure_gpu and idx > 10:
                    gpu_mem_used, gpu_memory_max , gpu_util = get_gpu_usage()
                    GPU_memory_meter_micro.update(gpu_mem_used)
                    GPU_util_meter_micro.update(gpu_util)
                    current_pid = os.getpid()
                    df = get_gpu_memory_map()
                    memory_used = df[df['pid'] == current_pid]['memory.used [MiB]'].values[0]
                    print(f"Memory used by current process: {memory_used} MiB")
                    measure_gpu = False

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Raw outputs of size {out.shape=}')

                out, y = self.patcher.unpatch(out, y)
                #Output encoding only works if output is stiched
                if output_encoder is not None and self.mg_patching_stitching:
                    out = output_encoder.decode(out)
                    y = output_encoder.decode(y)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. Processed (unpatched) outputs of size {out.shape=}')

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        loss = training_loss(out.float(), y)
                else:
                    if len(y.shape) == 2:
                        # todo: currently a bug with 1D Burgers, must be fixed.
                        y = y.view(y.shape[0], 1, y.shape[1])
                    loss = training_loss(out.float(), y)

                if regularizer:
                    loss += regularizer.loss

                loss.backward()


                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1
            time_meter.update(epoch_train_time)
            _, gpu_max_mem, _ = get_gpu_usage()
            GPU_memory_meter_macro.update(gpu_max_mem)
            GPU_util_meter_macro.update(GPU_util_meter_micro.avg)
            measure_gpu = True

            del x, y

            train_err/= n_train
            avg_loss /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 
                
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'
                msg_info = f'avg epoch time={time_meter.avg:.2f}, avg_gpu_mem/all_gpu_mem={GPU_memory_meter_macro.avg:.2f}/{gpu_mem_capacity:.2f}(GB), avg_gpu_util={GPU_util_meter_macro.avg:.2f}%'

                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

                for loader_name, loader in test_loaders.items():
                    if loader_name == 1024 and epoch < 499:
                        print('not final epoch, not evaluating 1024 res data')
                        continue
                    
                    if epoch == self.n_epochs - 1 and self.log_output:
                        to_log_output = True
                    else:
                        to_log_output = False

                    errors = self.evaluate(model, eval_losses, loader, output_encoder, log_prefix=loader_name)

                    for loss_name, loss_value in errors.items():
                        msg += f', {loss_name}={loss_value:.4f}'
                        values_to_log[loss_name] = loss_value

                if regularizer:
                    avg_lasso_loss /= self.n_epochs
                    msg += f', avg_lasso={avg_lasso_loss:.5f}'

                if self.verbose and is_logger:
                    print(msg)
                    print(msg_info)
                    sys.stdout.flush()

                # Wandb loging
                if self.wandb_log and is_logger:
                    for pg in optimizer.param_groups:
                        lr = pg['lr']
                        values_to_log['lr'] = lr
                    wandb.log(values_to_log, step=epoch, commit=True)

            if self.precision_schedule and epoch in self.precision_schedule:
                # todo: currently hard-coded for a schedule of half_prec_fourier, half_prec_inverse, full-precision
                # todo: also currently needs starting half_prec_fourier=True and stabilizer='tanh'
                fourier_precision = model.fourier_precision
                if fourier_precision[0]:
                    fourier_precision = (False, True)
                    model.fourier_precision = fourier_precision
                    print('set half_prec_fourier:', model.fno_blocks.convs.half_prec_fourier, 'half_prec_inverse:',model.fno_blocks.convs.half_prec_inverse, 'amp', self.amp_autocast)
                elif fourier_precision[1]:
                    fourier_precision = (False, False)
                    model.fourier_precision = fourier_precision
                    self.amp_autocast = False
                    print('set half_prec_fourier:', model.fno_blocks.convs.half_prec_fourier, 'half_prec_inverse:',model.fno_blocks.convs.half_prec_inverse, 'amp', self.amp_autocast)
            
            #save model every save_interval epochs; contains model and checkpoint states 
            if epoch % self.save_interval == 0:
                self.save_model_checkpoint(-1, model, optimizer)
                if self.wandb_log and is_logger:
                    datestr = datetime.today().strftime('%Y-%m-%d')
                    save_path = os.path.join(self.model_save_dir, f'checkpoint_last_{datestr}.pt')
                    wandb.save(save_path)
                
        return 

    def evaluate(self, model, loss_dict, data_loader, output_encoder=None,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        model : model to evaluate
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        output_encoder : used to decode outputs if not None
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        model.eval()

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']

                n_samples += x.size(0)
                
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                x = x.to(self.device)
                
                out = model(x)
        
                out, y = self.patcher.unpatch(out, y, evaluation=True)
                if len(y.shape) == 2:
                    # todo: currently a bug with 1D Burgers, must be fixed.
                    y = y.view(y.shape[0], 1, y.shape[1])

                if output_encoder is not None:
                    out = output_encoder.decode(out)

                if (it == 0) and self.log_output and self.wandb_log and is_logger:
                    if out.ndim == 2:
                        img = out
                    else:
                        img = out.squeeze()[0]
                    wandb.log({f'image_{log_prefix}': wandb.Image(img.unsqueeze(-1).cpu().numpy())}, commit=False)
                
                for loss_name, loss in loss_dict.items():
                    errors[f'{log_prefix}_{loss_name}'] += loss(out, y).item()

        del x, y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors

    def save_model_checkpoint(self, epoch, model, optimizer):
        """Saves a model checkpoint
        
        Parameters
        ----------
        epoch : int
            epoch number
        model : model to save
        optimizer : optimizer to save
        """
        datestr = datetime.today().strftime('%Y-%m-%d')
        if epoch == -1:
            save_path = os.path.join(self.model_save_dir, f'checkpoint_best_{datestr}.pt')
        else:
            save_path = os.path.join(self.model_save_dir, f'checkpoint_{epoch}_{datestr}.pt')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

        return 

    def load_model_checkpoint(self, epoch, model, optimizer, load_path=None):
        """Loads a model checkpoint
        
        Parameters
        ----------
        epoch : int
            epoch number
        model : model to load
        optimizer : optimizer to load
        """
        if not load_path:
            if epoch == -1:
                load_path = os.path.join(self.model_save_dir, f'checkpoint_best.pt')
            else:
                load_path = os.path.join(self.model_save_dir, f'checkpoint_{epoch}.pt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']
