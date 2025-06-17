import math
import matplotlib
import os
import torch

from enum import Enum
from matplotlib import pyplot as plt
from torch import Tensor, nn
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from lib import config_utils, data_utils, utils, visutils
from lib.models import MODELS
from lib.visutils import COLORMAPS

from SeqDiffusionPipeline import SeqDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler


class Predict_pipeline:
    def __init__(self, model, scheduler, device):
        self.pipeline = SeqDiffusionPipeline(model=model, scheduler=scheduler)

    def __call__(
            self,
            image = None,  # [B, T, C, H, W]
            mask = None,   # [B, T, 1 ,H, W]
            batch_positions = None,
            cond = None,
            generator = torch.manual_seed(0),
            eta: float = 0.0,
            num_inference_steps: int = 1,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "tensor",  # "pil", "numpy", "tensor"
            return_dict: bool = False
    ):
        # Sample gaussian noise to begin loop
        output = self.pipeline(image, mask, batch_positions, cond, generator, eta, num_inference_steps, use_clipped_model_output, output_type, return_dict)
        return output


class Method(Enum):
    SDT = 'SDT'


class Mode(Enum):
    LAST = 'last'
    NEXT = 'next'
    CLOSEST = 'closest'
    LINEAR_INTERPOLATION = 'linear_interpolation'
    NONE = None


class Imputation:
    def __init__(
            self,
            config_file_train: str | None,
            method: Literal['trivial', 'SDT'] = 'SDT',
            mode: Literal['last', 'next', 'closest', 'linear_interpolation'] | None = None,
            checkpoint: str | None = None,
            multigpus: True | False = False,
            num_inference_steps: int = 1,
            ifDate: bool = False,
            ifCond: bool = False,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):

        self.method = Method(method)
        self.mode = Mode(mode)
        self.checkpoint = checkpoint
        self.config_file_train = config_file_train
        self.multigpus = multigpus
        self.num_inference_steps = num_inference_steps
        self.ifDate = ifDate
        self.ifCond = ifCond

        if self.method == Method.SDT:
            if self.checkpoint is None:
                raise ValueError('No checkpoint specified.\n')

            if self.config_file_train is None:
                raise ValueError('No training configuration file specified.\n')

            if not os.path.isfile(self.config_file_train):
                raise FileNotFoundError(
                    f'Cannot find the configuration file used during training: {self.config_file_train}\n')

            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(f'Cannot find the model weights: {self.checkpoint}\n')

            # Read the configuration file used during training
            self.config = config_utils.read_config(self.config_file_train)

            # Extract the temporal window size and the number of channels used during training
            self.temporal_window = self.config.data.max_seq_length
            self.num_channels = data_utils.get_dataset(self.config, phase=self.config.misc.run_mode).num_channels

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        _ = torch.set_grad_enabled(False)

        # Get the model
        if self.method == Method.SDT:
            self.model, _ = utils.get_model(self.config, self.num_channels)
            self._resume()
            self.model.to(self.device).eval()

        # Get the Scheduler
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, prediction_type=self.config.training_settings.prediction_type)
        self.pipeline = Predict_pipeline(model=self.model, scheduler=scheduler, device="cuda")
        self.generator = generator

    def impute_sample(
            self,
            batch: Dict[str, Any],
            t_start: Optional[int] = None,
            t_end: Optional[int] = None,
            return_att: Optional[bool] = False
    ) -> Tuple[Dict[str, Any], Tensor, Tensor] | Tuple[Dict[str, Any], Tensor]:

        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch['x'] = batch['x'][:, t_start:t_end, ...]

            for key in ['y', 'masks', 'cloud_mask', 'masks_valid_obs']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]

            for key in ['days', 'position_days']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        # implement impute_sequence for SDT model
        if isinstance(self.model, MODELS['SDT']):
            batch = data_utils.to_device(batch, self.device)
            y_pred = impute_sequence_SDT(self.model, batch, self.temporal_window, self.pipeline, self.num_inference_steps, ifDate=self.ifDate, ifCond=self.ifCond)
            batch = data_utils.to_device(batch, 'cpu')
            y_pred = y_pred.cpu()

        return batch, y_pred

    def _resume(self) -> None:
        checkpoint = torch.load(self.checkpoint)
        if self.multigpus:
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint \'{self.checkpoint}\' loaded.')
        print(f"Chosen epoch: {checkpoint['epoch']}\n")
        del checkpoint


def impute_sequence_SDT(
        model, batch: Dict[str, Any], temporal_window: int, pipeline, num_inference_steps, ifDate: bool = False, ifCond: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Sliding-window imputation of satellite image time series using SDT.

    Assumption: `batch` consists of a single sample.
    """
    generator = torch.manual_seed(0)
    # for PASTIS dataset
    if ifCond:
        x = batch['x']
        cond = batch['cond']
    else:
        x = batch['x']
        cond = None

    if ifDate:
        date = batch['position_days']
    else:
        date = None

    mask = batch['masks']
    seq_length = x.shape[1]
    y_pred: Tensor
    att: Tensor

    # Pad the sequence with zeros
    if seq_length < temporal_window:
        pad = torch.zeros(
            (x.shape[0], temporal_window - x.shape[1], x.shape[2], x.shape[3], x.shape[4]), device=x.device
        )
        x = torch.cat([x, pad], dim=1)  # x and mask need to do the padding
        mask = torch.cat([mask, pad[:, :, :1, :, :]], dim=1)
        if date is not None:
            pad = torch.zeros((date.shape[0], temporal_window - date.shape[1]), device=date.device)
            date = torch.cat([date, pad], dim=1)
        if cond is not None:
            pad = torch.zeros(
                (cond.shape[0], temporal_window - cond.shape[1], cond.shape[2], cond.shape[3], cond.shape[4]),
                device=cond.device
            )
            cond = torch.cat([cond, pad], dim=1)
        y_pred = pipeline(x, mask, date, cond, generator=generator, num_inference_steps=num_inference_steps)
        y_pred = y_pred[:, :seq_length]

    elif seq_length == temporal_window:
        # Process the entire sequence in one go
        # y_pred = model(x, batch_positions=positions)
        y_pred = pipeline(x, mask, date, cond, generator=generator, num_inference_steps=num_inference_steps)

    else:
        t_start = 0
        t_end = temporal_window
        t_max = x.shape[1]
        cloud_coverage = torch.mean(batch['masks'], dim=(0, 2, 3, 4))
        reached_end = False

        while not reached_end:
            # y_pred_chunk = model(x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end])
            if date is not None:
                y_pred_chunk = pipeline(x[:, t_start:t_end], mask[:, t_start:t_end], date[:, t_start:t_end], cond[:, t_start:t_end], generator=generator, num_inference_steps=num_inference_steps)
            else:
                y_pred_chunk = pipeline(x[:, t_start:t_end], mask[:, t_start:t_end], batch_positions=None, cond=cond[:, t_start:t_end], generator=generator, num_inference_steps=num_inference_steps)


            if t_start == 0:
                # Initialize the full-length output sequence
                B, T, _, H, W = x.shape
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)

                y_pred[:, t_start:t_end] = y_pred_chunk

                # Move the temporal window
                t_start_old = t_start
                t_end_old = t_end
                t_start, t_end = move_temporal_window_next(t_start, t_max, temporal_window, cloud_coverage)
            else:
                # Find the indices of those frames that have been processed by both the previous and the current
                # temporal window
                t_candidates = torch.Tensor(
                    list(set(torch.arange(t_start_old, t_end_old).tolist()) & set(
                        torch.arange(t_start, t_end).tolist()))
                ).long().to(x.device)

                # Find the frame for which the difference between the previous and the current prediction is
                # the lowest:
                # use this frame to switch from the previous imputation results to the current imputation results
                error = torch.mean(
                    torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start]),
                    dim=(0, 2, 3, 4)
                )
                t_switch = error.argmin().item() + t_start
                y_pred[:, t_switch:t_end] = y_pred_chunk[:, (t_switch - t_start)::]

                if t_end == t_max:
                    reached_end = True
                else:
                    # Move the temporal window
                    t_start_old = t_start
                    t_end_old = t_end
                    t_start, t_end = move_temporal_window_next(
                        t_start_old, t_max, temporal_window, cloud_coverage
                    )

    return y_pred


def move_temporal_window_end(t_max: int, temporal_window: int) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation such that the last frame of the temporal window coincides with the
    last frame of the image sequence.

    Args:
        t_max:              int, sequence length of the image sequence
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    t_start = t_max - temporal_window
    t_end = t_max

    return t_start, t_end


def move_temporal_window_next(
        t_start: int, t_max: int, temporal_window: int, cloud_coverage: Tensor
) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation by half of the temporal window size (= stride).
    If the first frame within the new temporal window is cloudy (cloud coverage above 10%), the temporal window is
    shifted by at most half the stride (backward or forward) such that the first frame is as least cloudy as
    possible.

    Args:
        t_start:            int, frame index, start of the subsequence for processing
        t_max:              int, frame index, t_max - 1 is the last frame of the subsequence for processing
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing
        cloud_coverage:     torch.Tensor, (T,), cloud coverage [-] per frame

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    stride = temporal_window // 2
    t_start += stride

    if t_start + temporal_window > t_max:
        # Reduce the stride such that the end of the temporal window coincides with the end of the entire sequence
        t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    else:
        # Check if the start of the next temporal window is mostly cloud-free
        if cloud_coverage[t_start] <= 0.1:
            # Keep the default stride and ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)
        else:
            # Find the least cloudy frame within [t_start + stride - dt, t_start + stride + dt]
            dt = math.ceil(stride / 2)
            left = max(0, t_start - dt)
            right = min(t_start + dt + 1, t_max)

            # Frame(s) with the lowest cloud coverage within [t_start + stride - dt, t_start + stride + dt]
            t_candidates = (cloud_coverage[left:right] == cloud_coverage[left:right].min()).nonzero(as_tuple=True)[
                               0] + left

            # Take the frame closest to the standard stride
            t_start = t_candidates[torch.abs(t_candidates - t_start).argmin()].item()

            # Ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)

    return t_start, t_end





