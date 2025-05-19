import collections.abc
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F

from lib.datasets import DATASETS

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def to_device(sample: Dict, device: torch.device = torch.device('cuda')) -> Dict:
    sample_out = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sample_out[key] = val.to(device)
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.to(device))
                else:
                    new_val.append(e)
            sample_out[key] = new_val
        else:
            sample_out[key] = val

    return sample_out


def extract_sample(sample: Dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Union[float, int]]:
    inputs = sample['x']
    target = sample['y']
    masks = sample['masks']
    mask_valid = sample['masks_valid_obs'] if 'masks_valid_obs' in sample else None
    cloud_mask = sample['cloud_mask'] if 'cloud_mask' in sample else None
    # cloud_mask = sample['masks']
    indices_rgb = sample.get('c_index_rgb', torch.Tensor([2, 1, 0]))
    index_nir = sample.get('c_index_nir', torch.Tensor([np.nan]))

    if isinstance(indices_rgb[0], torch.Tensor):
        indices_rgb = indices_rgb[0]
    if not isinstance(index_nir, (float, int)):
        index_nir = index_nir[0].item()

    return inputs, target, masks, mask_valid, cloud_mask, indices_rgb, index_nir


def pad_tensor(x, l, pad_value=0):
    """
    Source: https://github.com/VSainteuf/utae-paps/blob/main/src/utils.py
    """

    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    """
    Modified version of: https://github.com/VSainteuf/utae-paps/blob/main/src/utils.py
    """

    elem = batch[0]

    elem_type = type(elem)

    if isinstance(elem, Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            #out = elem.new(storage)
            out = elem.new(storage).resize_(len(batch), *list(batch[0].size()))
        return torch.stack(batch, 0, out=out)
    if (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ in ("ndarray", "memmap"):
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f"Format not managed : {elem.dtype}")

            return pad_collate([torch.as_tensor(b) for b in batch])
        if elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    if isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    if isinstance(elem, int):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return batch
    if isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError(f"Format not managed : {elem_type}")


def get_dataloader(
        config: DictConfig, phase: str, pin_memory: bool = True, drop_last: bool = False,
        logger: Optional[logging.Logger] = None
) -> torch.utils.data.dataloader.DataLoader:
    """Returns a torch.utils.data.DataLoader instance."""

    dset = get_dataset(config, phase, logger)
    variable_seq_length = getattr(dset, 'variable_seq_length', False) and config.training_settings.batch_size > 1
    shuffle = config['misc']['run_mode'] != 'test'

    if variable_seq_length:
        collate_fn = lambda x: pad_collate(x, pad_value=config.method.pad_value)
    else:
        collate_fn = None

    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=config.training_settings.batch_size, shuffle=shuffle,
                                         num_workers=config.misc.num_workers, collate_fn=collate_fn,
                                         pin_memory=pin_memory, drop_last=drop_last)

    return loader


def get_dataset(config: DictConfig, phase: str, logger: Optional[logging.Logger] = None):
    """Returns a torch.utils.data.Dataset instance."""

    from lib.utils import without_keys

    assert config['misc']['run_mode'] in ['train', 'val', 'test']
    assert phase in ['train', 'val', 'train+val', 'test']

    if config.data.dataset not in DATASETS:
        if logger:
            logger.error(f'Unknown dataset: {config.data.dataset}\n')
        else:
            raise NotImplementedError(f'Unknown dataset: {config.data.dataset}\n')

    # Select the defined dataset
    Dataset = DATASETS[config.data.dataset]

    augment = phase == 'train'
    if 'hdf5_file' in config.data and isinstance(config.data.hdf5_file, DictConfig):
        # Choose the input hdf5 file depending on the phase
        dset = Dataset(
            hdf5_file=config.data.hdf5_file[phase], **without_keys(config.data, ['dataset', 'hdf5_file']),
            mask_kwargs=config.mask, augment=augment
        )
    else:
        dset = Dataset(**without_keys(config.data, ['dataset', 'preprocessed', 'render_occluded_above_p', 'return_cloud_mask']),
                       mask_kwargs=config.mask, augment=augment)

    return dset


def compute_false_color(x: Tensor, index_rgb: Tensor | List[int | float], index_nir: int | float) -> Tensor:
    """
    Returns the false color composite (NIR, R, G) for every time step of the input sequence or
    for the single input image.

    Args:
        x:           torch.Tensor, image time series, T x C x H x W (sequence) or C x H x W (single image).
        index_rgb:   list of int, indices of the RGB channels.
        index_nir:   int, index of the NIR channel.

    Returns:         torch.Tensor, false color composite (NIR, R, G), T x 3 x H x W (sequence) or
                     3 x H x W (single image).
    """

    if x.dim() == 4:
        return torch.stack((x[:, index_nir, ...], x[:, index_rgb[0], ...], x[:, index_rgb[1], ...]), dim=1)

    return torch.stack((x[index_nir, ...], x[index_rgb[0], ...], x[index_rgb[1], ...]), dim=1)
