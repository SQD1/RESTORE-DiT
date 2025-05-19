from typing import Any, Dict, Literal

import math
import torch
import torchgeometry as tgm
from prodict import Prodict
from torch import Tensor

from lib.data_utils import extract_sample


def compute_sam(predicted: Tensor, target: Tensor, units: Literal['deg', 'rad'] = 'rad') -> Tensor:
    """
    Computes the spectral angle mapper (SAM) averaged over all time steps and batch samples.

    Args:
        predicted:   torch.Tensor,  (n_frames x C x H x W).
        target:      torch.Tensor,  (n_frames x C x H x W).

    Returns:
        sam_value:   torch.Tensor, (1, ), mean spectral angle [rad].
    """

    dot_product = (predicted * target).sum(dim=1)
    predicted_norm = predicted.norm(dim=1)
    target_norm = target.norm(dim=1)

    # Compute the SAM score for all pixels with vector norm > 0
    flag = torch.logical_and(predicted_norm != 0., target_norm != 0.)
    if torch.any(flag):
        spectral_angles = torch.clamp(dot_product[flag] / (predicted_norm[flag] * target_norm[flag]), -1, 1).acos()
        sam_score = torch.mean(spectral_angles)

        if units == 'deg':
            sam_score *= 180/math.pi

        return sam_score
    else:
        return None


class EvalMetrics:
    """
    Computes the metrics used to monitor the training progress or for evaluation.
    """

    def __init__(self, args: Dict):
        self.args = args
        self.masked_metrics = args.get('masked_metrics', False)
        self.sam_units = args.get('sam_units', 'rad')


        # True to evaluate the metrics over all pixels and separately for occluded and observed input pixels;
        # False to evaluate the metrics over all pixels only
        self.eval_occluded_observed = args.get('eval_occluded_observed', False)

        # MAE (mean absolute error)
        self.mae = lambda predicted, target: torch.mean(torch.abs(predicted - target))

        # MSE (mean squared error)
        self.mse = lambda predicted, target: torch.mean(torch.square(predicted - target))

        # RMSE (root mean square error)
        self.rmse = lambda predicted, target: torch.sqrt(torch.mean(torch.square(predicted - target)))

        # SSIM (structural similarity index)
        self.dssim = tgm.losses.SSIM(5, reduction='mean')

        # PSNR (peak signal-to-noise ratio)
        self.psnr = lambda predicted, target: 20 * torch.log10(1 / self.rmse(predicted, target))

    def __call__(self, batch: Dict[str, Any], predicted: Tensor) -> Dict[str, float]:
        """
        Args:
            batch:           dict, batch sample; the following information will be extracted from the batch using
                             the functionality extract_sample():
                                target:       torch.Tensor, (B x T x C x W x H); target sequence.
                                masks:        torch.Tensor, (B x T x 1 x W x H); a pixel value of 0 indicates an
                                              observed (non-masked) input pixel and a pixel value of 1 a masked input
                                              pixel.
                                mask_valid:   torch.Tensor, (B, T); 0 indicates a non-valid time step (zero-padded) and
                                              1 a valid time step.
                                cloud_mask:   torch.Tensor, (B x T x 1 x W x H), 0 indicates a non-occluded target pixel
                                              and 1 an occluded target pixel.
                                indices_rgb:  list of three int, indices of the RGB channels.
                                index_nir:    int, index of the NIR channel.
            predicted:       torch.Tensor, (B x T x C x W x H); predicted sequence.
        """

        _, target, masks, mask_valid, cloud_mask, _, _ = extract_sample(batch)

        if cloud_mask is None:
            self.masked_metrics = False
            self.prefix = ''
        else:
            self.prefix = 'masked_'

        # Initialize metrics
        metrics = Prodict()

        # Concatenate batch and time dimension
        B, T, C, H, W = predicted.shape
        n_frames = B * T
        predicted = predicted.view(n_frames, C, H, W)
        target = target.view(n_frames, C, H, W)
        masks = masks.view(n_frames, 1, H, W).expand(target.shape)

        # Evaluate valid and non-padded frames only
        if mask_valid is not None:
            mask_valid = mask_valid.view(n_frames).bool()
            predicted = predicted[mask_valid, ...]
            target = target[mask_valid, ...]
            masks = masks[mask_valid, ...]

        # Structural similarity index (SSIM) evaluated over all images
        if self.args.get('ssim', False):
            dssim = self.dssim(predicted, target)  # outputs (1 - SSIM)/2; structural dissimilarity
            metrics['ssim'] = 1 - 2 * dssim

            # Structural similarity index (SSIM) evaluated over all images with data gaps
            if self.eval_occluded_observed:
                occ_images = (masks == 1.).any(dim=-1).any(dim=-1).any(dim=-1)
                metrics['ssim_images_occluded_input_pixels'] = 1 - 2 * self.dssim(predicted[occ_images], target[occ_images])
                metrics['ssim_images_observed_input_pixels'] = 1 - 2 * self.dssim(predicted[~occ_images], target[~occ_images])

        # if self.masked_metrics == False: metrics are computed over all output pixels
        # if self.masked_metrics == True: metrics are computed over all non-occluded target pixels (according to GT cloud masks)
        # print(self.masked_metrics)
        if self.masked_metrics:
            cloud_mask = cloud_mask.view(n_frames, 1, H, W)

            if mask_valid is not None:
                n_frames = target.shape[0]
                cloud_mask = cloud_mask[mask_valid, ...]

            # Evaluate non-occluded target pixels only
            flag = cloud_mask.permute(0, 2, 3, 1).reshape(n_frames * H * W) == 0.

            # Tensor shapes: (n_frames * H * W, C)
            predicted = predicted.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            target = target.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            masks = masks.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]

        # MAE (mean absolute error) evaluated over all pixels in the input sequence
        if self.args.get('mae', False):
            metrics[f'{self.prefix}mae'] = self.mae(predicted, target)

            if self.eval_occluded_observed:
                metrics[f'{self.prefix}mae_occluded_input_pixels'] = self.mae(predicted[masks == 1.], target[masks == 1.])
                print("see", metrics[f'{self.prefix}mae_occluded_input_pixels'])
                metrics[f'{self.prefix}mae_observed_input_pixels'] = self.mae(predicted[masks == 0.], target[masks == 0.])

        # Root mean squared error (RMSE)
        if self.args.get('rmse', False):
            metrics[f'{self.prefix}rmse'] = self.rmse(predicted, target)

            if self.eval_occluded_observed:
                metrics[f'{self.prefix}rmse_occluded_input_pixels'] = self.rmse(predicted[masks == 1.], target[masks == 1.])
                metrics[f'{self.prefix}rmse_observed_input_pixels'] = self.rmse(predicted[masks == 0.], target[masks == 0.])

        # Mean squared error (MSE)
        if self.args.get('mse', False):
            metrics[f'{self.prefix}mse'] = self.mse(predicted, target)

            if self.eval_occluded_observed:
                metrics[f'{self.prefix}mse_occluded_input_pixels'] = self.mse(predicted[masks == 1.], target[masks == 1.])
                metrics[f'{self.prefix}mse_observed_input_pixels'] = self.mse(predicted[masks == 0.], target[masks == 0.])

        # PSNR
        if self.args.get('psnr', False):
            metrics[f'{self.prefix}psnr'] = self.psnr(predicted, target)

            if self.eval_occluded_observed:
                metrics[f'{self.prefix}psnr_occluded_input_pixels'] = self.psnr(predicted[masks == 1.], target[masks == 1.])
                metrics[f'{self.prefix}psnr_observed_input_pixels'] = self.psnr(predicted[masks == 0.], target[masks == 0.])

        # SAM
        if self.args.get('sam', False):
            if self.masked_metrics:
                # Introduce a batch and a second spatial dimension to comply with the data structure expected by
                # compute_sam():
                # (n_frames * H * W, C) -> (C, n_frames * H * W) -> (1, C, n_frames * H * W, 1)
                predicted = predicted.permute(1, 0).unsqueeze(0).unsqueeze(3)
                target = target.permute(1, 0).unsqueeze(0).unsqueeze(3)
            sam = compute_sam(predicted, target, units=self.sam_units)
            if sam is not None:
                metrics[f'{self.prefix}sam'] = sam

            if self.eval_occluded_observed:
                if cloud_mask is None:
                    predicted = predicted.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)
                    target = target.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)
                    masks = masks.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)

                    predicted = predicted.permute(1, 0).unsqueeze(0).unsqueeze(3)
                    target = target.permute(1, 0).unsqueeze(0).unsqueeze(3)

                sam = compute_sam(
                    predicted[:, :, (masks == 1.).all(dim=1), :], target[:, :, (masks == 1.).all(dim=1), :],
                    units=self.sam_units
                )
                if sam is not None:
                    metrics[f'{self.prefix}sam_occluded_input_pixels'] = sam

                sam = compute_sam(
                    predicted[:, :, (masks == 0.).all(dim=1), :], target[:, :, (masks == 0.).all(dim=1), :],
                    units=self.sam_units
                )
                if sam is not None:
                    metrics[f'{self.prefix}sam_observed_input_pixels'] = sam

        for key, value in metrics.items():
            metrics[key] = value.item()

        return metrics
