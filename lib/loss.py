from typing import Any, Dict, Tuple

import torch
import torchgeometry as tgm
from prodict import Prodict
from torch import Tensor, nn

from lib.data_utils import extract_sample


class TrainLoss:
    """
    Loss computation.

    Args:
        args: dict, parameters that define the type of loss functions and their relative weighting.
    """

    def __init__(self, args: Dict):
        self.args = args

        # L1 reconstruction loss
        self.recon_l1 = nn.L1Loss(reduction='mean')

        # SSIM loss: (1 - SSIM)/2
        self.ssim = tgm.losses.SSIM(5, reduction='mean')

        # Weights
        self.weights = Prodict()
        self.weights.l1_loss = self.args.get('l1_loss_w', 1.0)
        self.weights.l1_loss_occluded_input_pixels = self.args.get('l1_loss_occluded_input_pixels_w', 1.0)
        self.weights.l1_loss_observed_input_pixels = self.args.get('l1_loss_observed_input_pixels_w', 1.0)
        self.weights.ssim_loss = self.args.get('ssim_loss_w', 1.0)
        self.weights.masked_l1_loss = self.args.get('masked_l1_loss_w', 1.0)

    def __call__(self, batch: Dict[str, Any], predicted: Tensor) -> Tuple[Dict[str, float], Tensor]:
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
            predicted:       torch.Tensor, (B x T x C x W x H); predicted sequence.
        """

        _, target, masks, mask_valid, cloud_mask, _, _ = extract_sample(batch)

        # Initialize losses
        loss_dict = Prodict()

        # Concatenate batch and time dimension
        b, t, c, h, w = predicted.shape
        predicted = predicted.view(b * t, c, h, w)
        target = target.view(b * t, c, h, w)
        masks = masks.view(b * t, 1, h, w).expand(target.shape)

        # Evaluate valid and non-padded frames only
        if mask_valid is not None:
            mask_valid = mask_valid.view(b*t).bool()
            predicted = predicted[mask_valid, ...]
            target = target[mask_valid, ...]
            masks = masks[mask_valid, ...]
            if cloud_mask is not None:
                cloud_mask = cloud_mask.view(b * t, 1, h, w)[mask_valid, ...].expand(target.shape)

        # L1 reconstruction loss w.r.t. all pixels
        if self.args.get('l1_loss', False) and self.weights.l1_loss:
            loss_dict.l1_loss = self.recon_l1(predicted, target)

        # L1 reconstruction loss w.r.t. masked input pixels
        if self.args.get('l1_loss_occluded_input_pixels', False) and self.weights.l1_loss_occluded_input_pixels > 0:
            loss_dict.l1_loss_occluded_input_pixels = self.recon_l1(predicted[masks == 1.], target[masks == 1.])

        # L1 reconstruction loss w.r.t. observed input pixels (unmasked input pixels)
        if self.args.get('l1_loss_observed_input_pixels', False) and self.weights.l1_loss_observed_input_pixels > 0:
            loss_dict.l1_loss_observed_input_pixels = self.recon_l1(predicted[masks == 0.], target[masks == 0.])

        # SSIM loss
        if self.args.get('ssim_loss', False) and self.weights.ssim_loss > 0:
            loss_dict.ssim_loss = self.ssim(predicted, target)

        # L1 reconstruction loss w.r.t. all pixels associated with a valid ground truth reflectance
        if self.args.get('masked_l1_loss', False) and self.weights.masked_l1_loss:
            loss_dict.masked_l1_loss = self.recon_l1(predicted[cloud_mask == 0.], target[cloud_mask == 0.])

        # Compute the total loss as the sum of (weighted) individual losses
        total_loss = torch.zeros(1, requires_grad=True).to(target.device)
        for key in loss_dict:
            total_loss += (loss_dict[key] * self.weights[key])
            loss_dict[key] = loss_dict[key].detach().item()

        loss_dict.total_loss = total_loss.detach().item()

        return loss_dict, total_loss
