import logging
from dataclasses import dataclass
import time
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, Dict, Optional, Tuple
from torch import Tensor
from PIL import Image
import numpy as np
import prodict
from prodict import Prodict
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from lib import logger, visutils
from lib.logger import AverageMeter
from lib.loss import TrainLoss
from lib.data_utils import to_device
import wandb

from tqdm.auto import tqdm
from pathlib import Path
import os


def seconds_to_dd_hh_mm_ss(seconds_elapsed: int) -> Tuple[int, int, int, int]:
    days = seconds_elapsed // (24 * 3600)
    seconds_remainder = seconds_elapsed % (24 * 3600)
    hours = seconds_remainder // 3600
    seconds_remainder %= 3600
    minutes = seconds_remainder // 60
    seconds_remainder %= 60
    seconds = seconds_remainder

    return days, hours, minutes, seconds


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(output_dir, epoch, pipeline, val_batch):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = 1,
        generator=torch.manual_seed(0),
    ).images

    output = pipeline(val_batch['y'], val_batch['masks'], generator=torch.manual_seed(0)).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

class Trainer:
    def __init__(
            self,
            args: DictConfig,
            train_loader: torch.utils.data.dataloader.DataLoader,
            val_loader: torch.utils.data.dataloader.DataLoader,
            model,
            noise_scheduler,
            optimizer,
            scheduler
    ):
        self.args = args
        self.use_wandb = bool('wandb' in args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.CTHW = self.args.data.ifCTHW

        self.dataloader = {'train': train_loader, 'val': val_loader}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)
        self.args.accum_iter = self.args.get('accum_iter', 1)  # accumulate gradients for `accum_iter` iterations

        self.compute_losses = TrainLoss(self.args.loss)
        self.loss_fn = F.mse_loss
        # self.compute_metrics = EvalMetrics(self.args.metrics)

        # Losses: Initialize statistics
        self.train_stats = self._stats_meter(stats_type='loss')
        self.val_stats = self._stats_meter(stats_type='loss')

        # Losses: Initialize metrics
        self.train_metrics = self._stats_meter(stats_type='metrics')
        self.val_metrics = self._stats_meter(stats_type='metrics')

        self.best_loss = np.inf
        self.epoch_best_loss = np.nan

        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.args.path_model_best = os.path.join(self.args.checkpoint_dir, 'Model_best.pth')
        self.args.path_model_last = os.path.join(self.args.checkpoint_dir, 'Model_last.pth')
        self.logger = logger.prepare_logger('train_logger', level=logging.INFO, log_to_console=True,
                                            log_file=os.path.join(args.save_dir, 'training.log'))

        self.noise_scheduler = noise_scheduler

        # Set up wandb
        if self.use_wandb:
            os.makedirs(self.args.wandb.dir, exist_ok=True)
            wandb.init(**self.args.wandb, settings=wandb.Settings(start_method="fork"))
            wandb.config.update(OmegaConf.to_container(self.args))
            self.writer = None

            # Define the wandb summary metrics
            # for key, value in self.args.metrics.items():
            #     if key == 'masked_metrics':
            #         pass
            #     elif value:
            #         wandb.define_metric(f"train_metrics/{key}", summary=OBJECTIVE[key])
            #         wandb.define_metric(f"val_metrics/{key}", summary=OBJECTIVE[key])

            # wandb.define_metric('train/total_loss', summary=OBJECTIVE['total_loss'])
            # wandb.define_metric('val/total_loss', summary=OBJECTIVE['total_loss'])
        else:
            os.makedirs(os.path.join(self.args.save_dir, 'tb'), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, 'tb'))

        # Resume training
        if self.args.resume and self.args.pretrained_path:
            self._resume(path=self.args.pretrained_path)
        else:
            self.logger.info('\nTraining from scratch.\n')
            self.epoch = 0
            self.iter = 0

    def _stats_meter(self, stats_type: str) -> prodict.Prodict:
        meters = Prodict()
        stats = self._stats_dict(stats_type)
        for key, _ in stats.items():
            meters[key] = AverageMeter()

        return meters

    def _get_lr(self, group: int = 0) -> float:
        return self.optimizer.param_groups[group]['lr']

    def _resume(self, path: str) -> None:
        """
        Resumes training.

        Args:
            path:  str, path of the pretrained model weights.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(f'No checkpoint found at {path}\n')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        # if self.args.get('load_scheduler_state_dict', True) and 'scheduler_state_dict' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #
        # # Extract the last training epoch
        # self.epoch = checkpoint['epoch'] + 1
        # self.iter = checkpoint['iter']
        # self.args.num_epochs += self.epoch
        #
        # # Best validation loss so far
        # self.best_loss = checkpoint['best_loss']
        # self.epoch_best_loss = checkpoint['epoch']
        #
        # self.logger.info('\n\nRestoring the pretrained model from epoch %d.', self.epoch - 1)
        # self.logger.info('Successfully loaded pretrained model weights from %s.\n', path)
        # self.logger.info('Current best loss %.4f\n', self.best_loss)

        self.epoch = 0
        self.iter = 0

    def _log_iter_epoch(self) -> None:
        if self.use_wandb:
            wandb.log({'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('epoch', self.epoch, self.iter)

    def _log_learning_rate(self) -> None:
        if self.use_wandb:
            wandb.log({'log_lr': np.log10(self._get_lr()), 'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('log_lr', np.log10(self._get_lr()), self.epoch)

    def _stats_dict(self, stats_type: str) -> prodict.Prodict:
        stats = Prodict()

        if stats_type == 'metrics':
            masked_metrics = self.args.metrics.masked_metrics
            for key, value in self.args.metrics.items():
                if key == 'masked_metrics':
                    pass
                elif value:
                    if masked_metrics and key != 'ssim':
                        stats[f'masked_{key}'] = np.inf
                    else:
                        stats[key] = np.inf

        elif stats_type == 'loss':
            for key, value in self.args.loss.items():
                # Exclude weight keys
                if value and isinstance(value, bool):
                    stats[key] = np.inf
            stats.total_loss = np.inf

        return stats

    def _save_checkpoint(self, filepath: str) -> None:
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.epoch_best_loss
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def _log_stats_meter(self, phase: str) -> None:
        if self.use_wandb:
            if phase == 'train':
                wandb.log({
                    'train_losses/' + k: v.avg for k, v in self.train_stats.items()
                }, step=self.iter)
            else:
                stats = {'val_losses/' + k: v.avg for k, v in self.val_stats.items()}
                stats['epoch'] = self.epoch
                wandb.log(stats, step=self.iter)

        else:
            if phase == 'train':
                for k, v in self.train_stats.items():
                    self.writer.add_scalar('train_losses/' + k, v.avg, self.iter)
            else:
                for k, v in self.val_stats.items():
                    self.writer.add_scalar('val_losses/' + k, v.avg, self.iter)

    def train(self) -> None:
        # Log gradients and model parameters
        if self.use_wandb and self.args.get('log_gradients', False):
            wandb.watch(self.model, log='all')

        self.logger.info('\nStart training...\n')
        start_time = time.time()

        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_description("Epoch")
            tnr.set_postfix(epoch=self.epoch, training_loss=np.nan)
            for _ in tnr:
                if self.scheduler is not None:
                    self._log_learning_rate()

                # -------------------------------- TRAINING -------------------------------- #
                self.train_epoch(tnr)

                # After the epoch if finished, update the learning rate scheduler
                if self.scheduler is not None:
                    self._log_learning_rate()

                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(self.val_stats.total_loss.avg)
                    else:
                        self.scheduler.step()

                # Save the model at the selected interval and validate the model
                if (self.epoch + 1) % self.args.checkpoint_every_n_epochs == 0:
                    name = 'Model_after_' + str(self.epoch + 1) + '_epochs.pth'
                    self._save_checkpoint(os.path.join(self.args.checkpoint_dir, name))



                self.epoch += 1

        time_elapsed = int(time.time() - start_time)
        self.logger.info(
            '\n\nTraining finished!\nTraining time: %dd %dh %dm %ds' % seconds_to_dd_hh_mm_ss(time_elapsed))
        self.logger.info('\nBest model at epoch: %d', self.epoch_best_loss)
        self.logger.info(f'Validation loss of the best model: {self.best_loss:.4f}')

        # Save the last model
        self._save_checkpoint(self.args.path_model_last)

        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.train_stats = self._stats_meter(stats_type='loss')
        # self.train_metrics = self._stats_meter(stats_type='metrics')
        self.model.train()

        # Clear gradients
        for param in self.model.parameters():
            param.grad = None

        with tqdm(self.dataloader['train'], leave=False) as tnr_train:
            tnr_train.set_description("Training")
            tnr_train.set_postfix(epoch=self.epoch, training_loss=np.nan, best_loss=self.best_loss)

            for i, batch in enumerate(tnr_train):
                self._log_iter_epoch()
                loss = self.inference_one_batch(batch)
                loss_dict = Prodict()
                loss_dict.l1_loss_occluded_input_pixels = loss.detach().item()

                # Update to stats_meter
                for key, value in loss_dict.items():
                    self.train_stats[key].update(value)

                loss = loss / self.args.accum_iter
                loss.backward()

                # gradient accumulation
                if ((i + 1) % self.args.accum_iter == 0) or (i + 1 == len(self.dataloader['train'])):
                    # Gradient clipping
                    if getattr(self.args, 'gradient_clip_norm', False) and self.args.gradient_clip_norm > 0.:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)

                    elif getattr(self.args, 'gradient_clip_value', False) and self.args.gradient_clip_value > 0.:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_value)

                    self.optimizer.step()

                    # Clear gradients
                    for param in self.model.parameters():
                        param.grad = None

                # if (i + 1) % min(self.args.logstep_train, len(self.dataloader['train'])) == 0:
                #     self._log_stats_meter(phase='train')
                #
                #     tnr_train.set_postfix(epoch=self.epoch,
                #                           training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg
                #                           )
                #     if tnr is not None:
                #         tnr.set_postfix(epoch=self.epoch,
                #                         training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg
                #                         )

                self.iter += 1

            tnr_train.set_postfix(epoch=self.epoch,
                                  training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg)

            if tnr is not None:
                tnr.set_postfix(epoch=self.epoch,
                                training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg,
                                best_loss=self.best_loss)


            self.logger.info((f'Train:\tEpoch: {self.epoch}\t' + f'learning rate: {self._get_lr():.8f}\t'
                              ''.join([f'{k}: {v.avg:.6f}\t' for k, v in self.train_stats.items()])))

            if self.best_loss > self.train_stats.l1_loss_occluded_input_pixels.avg:
                self.best_loss = self.train_stats.l1_loss_occluded_input_pixels.avg
                self.epoch_best_loss = self.epoch
                self._save_checkpoint(self.args.path_model_best)

            # Reset stats and metrics
            for key in self.train_stats:
                self.train_stats[key].reset()



    def inference_one_batch(
            self, batch: Dict[str, Any]
    ) -> Tensor:

        # batch = to_device(batch, self.device)
        y_0 = batch['y'].cuda()                #  (B, T, C, H, W)
        mask = batch['masks'].cuda()           #  (B, T, 1, H, W)
        date = batch['position_days'].cuda()            #  (B, T)           'position_days' or None
        # date = None
        cond = batch['cond'].cuda()            #  (B, T, 3, H, W)
        bs, length = y_0.shape[0], y_0.shape[1]

        if self.CTHW:
            y_0 = y_0.permute(0, 2, 1, 3, 4)     #  (B, C, T, H, W)
            mask = mask.permute(0, 2, 1, 3, 4)   #  (B, 1, T, H, W)

        # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision

        noise = torch.randn(y_0.shape).to(y_0.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=y_0.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(y_0, noise, timesteps)

        pred_hat = self.model(noisy_images * mask + (1. - mask) * y_0, timesteps, date=date, cond=cond)  # date = None
        # noise_hat = self.model(noisy_images * mask + (1. - mask) * y_0, timesteps, batch_positions=batch['position_days'])

        loss = self.loss_fn(mask * y_0, mask * pred_hat)

        return loss
