import glob
import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchinfo
from omegaconf import DictConfig, OmegaConf

from lib.models import MODELS
from lib.models.weight_init import weight_init

from lib.trainerDiff_SARv4 import Trainer  #  Final version



from diffusers.optimization import get_cosine_schedule_with_warmup


def create_output_directory(config: DictConfig) -> str:
    """
    Creates the output directory.

    Args:
        config:             dict, yaml configuration file imported as dictionary.

    Returns:
        output_directory:   str, path of the output directory.
    """

    if 'output' in config and 'output_directory' in config.output and isinstance(config.output.output_directory, str):
        os.makedirs(config.output.output_directory, exist_ok=True)

        if 'suffix' in config.output and isinstance(config.output.suffix, str):
            # The name of the output directory is the current date and time, followed by a suffix defined in the
            # configuration file
            name = datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + config.output.suffix
        else:
            # The name of the output directory is the current date and time without suffix
            name = datetime.now().strftime('%Y-%m-%d_%H-%M')

        output_directory = os.path.join(config.output.output_directory, name)
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = None

    return output_directory


def count_model_parameters(model) -> int:
    """
    Counts the number of trainable parameters in a torch model.

    Args:
        model:   nn.Module instance, input model.

    Returns:
        int, number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_default_model_settings(model, args_model: DictConfig) -> None:
    """
    Populates `args_model` (in-place) with the default parameter settings of `model`, provided that the parameter is
    not yet specified in `args_model`.

    Args:
        model:      nn.Module, the model used for training.
        args_model: dict, dictionary storing the model architecture parameters (extracted from config file).
    """

    default_parms = {}

    if isinstance(model, MODELS['utilise']):
        default_parms = ['encoder_widths', 'decoder_widths', 'str_conv_k', 'str_conv_s', 'str_conv_p', 'agg_mode',
                         'upconv_type', 'encoder_norm', 'decoder_norm', 'skip_norm', 'activation', 'n_head', 'd_k',
                         'bias_qk', 'attn_dropout', 'dropout', 'return_maps', 'padding_mode', 'skip_attention',
                         'output_activation', 'n_groups', 'dim_per_group', 'group_norm_eps', 'ltae_norm',
                         'str_conv_k_up', 'str_conv_p_up', 'norm_first']

    for param in default_parms:
        if param not in args_model:
            val = getattr(model, param)
            args_model[param] = val.value if isinstance(val, Enum) else val


def get_model(config: DictConfig, input_dim: int, logger: Optional[logging.Logger] = None):
    """
    Returns a model instance and its parameter settings.

    Args:
        config:     dict, yaml configuration file imported as dictionary.
        input_dim:  int, number of input channels.
        logger:     logger instance (if None, output is print to console).

    Returns:
        model:      nn.Module, model to be used for training.
        args_model: dict, dictionary storing the model architecture parameters.
    """

    model_type = config.method.model_type

    if model_type not in MODELS or model_type not in config:
        if logger is not None:
            logger.error(f"{model_type} model is not implemented.\n")
        else:
            raise NotImplementedError(f"ERROR: {model_type} model is not implemented.\n")

    args_model = deepcopy(config[model_type]) if model_type in config else OmegaConf.create()

    model = MODELS[model_type](**args_model)

    # Collect default values (if not specified in config) in order to log them in wandb
    get_default_model_settings(model, args_model)

    return model, args_model


def get_optimizer(config: DictConfig, model, logger: Optional[logging.Logger] = None):
    """
    Returns an optimizer instance.

    Args:
        config:      dict, yaml configuration file imported as dictionary.
        model:       nn.Module instance, model to be used for training.
        logger:      logger instance (if None, output is print to console).

    Returns:
        optimizer:   torch.optim.optimizer instance, optimizer to be used for training.
    """

    if config.optimizer.name == 'Adam':
        betas = config.optimizer.get('betas', (0.9, 0.999))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay,
            betas=betas
        )
    elif config.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.optimizer.learning_rate,  weight_decay=config.optimizer.weight_decay,
            momentum=config.optimizer.momentum
        )
    else:
        if logger is not None:
            logger.error(f"{config.optimizer.name} optimizer is not implemented.\n")
            sys.exit(1)
        else:
            raise NotImplementedError(f"ERROR: {config.optimizer.name} optimizer is not implemented.\n")

    return optimizer


def get_scheduler(config: DictConfig, optimizer, len_trainset, logger: Optional[logging.Logger] = None):
    """
    Returns a learning rate scheduler instance.

    Args:
        config:    dict, yaml configuration file imported as dictionary.
        optimizer: torch.optim.optimizer instance, optimizer to be used for training.
        len_trainset: length of training dataset.
        logger:    logger instance (if None, output is print to console).

    Returns:
        scheduler: torch.optim.lr_scheduler instance, learning rate scheduler to be used for training
                   (None, if the learning rate scheduler is disabled).
    """

    if config.scheduler.enabled:
        name = config.scheduler.name
        settings = without_keys(config.scheduler, ['name', 'enabled'])

        if name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, **settings)
        elif name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, verbose=False, **settings)
        elif name == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **settings)
        elif name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=False, **settings)
        elif name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, verbose=False, T_max=config.training_settings.num_epochs
            )
        elif name == 'CosineWithWarmup':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.optimizer.lr_warmup_steps,
                num_training_steps=(len_trainset * config.training_settings.num_epochs),
            )
        else:
            if logger:
                logger.error(f"{name} learning rate scheduler is not implemented.\n")
                sys.exit(1)
            else:
                raise NotImplementedError(f"ERROR: {name} learning rate scheduler is not implemented.\n")
    else:
        scheduler = None

    return scheduler


def get_trainer(
        config: DictConfig, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        model, noise_scheduler, optimizer, scheduler
) -> Trainer:
    """
    Returns a Trainer instance.

    Args:
        config:          dict, json configuration file imported as dictionary.
        train_loader:    torch.utils.data.DataLoader instance, training data.
        val_loader:      torch.utils.data.DataLoader instance, validation data.
        model:           nn.Module instance, model to be used for training.
        noise_scheduler: Strategy of adding noises. e.g. DDIMScheduler
        optimizer:       torch.optim.optimizer instance, optimizer to be used for training.
        scheduler:       torch.optim.lr_scheduler instance, learning rate scheduler to be used for training
                         (None, if the learning rate scheduler is disabled).

    Returns:
        instance of the Trainer class.
    """

    # Prepare configuration file for logging
    args = without_keys(config, ['scheduler', 'training_settings', 'misc', 'output'])
    if not isinstance(args, DictConfig):
        args = OmegaConf.create(args)

    if not config.scheduler.enabled:
        args.scheduler = OmegaConf.create()
        args.scheduler.name = config.scheduler.name
        args.scheduler.enabled = config.scheduler.enabled
    else:
        args.scheduler = deepcopy(getattr(config, 'scheduler'))

    for key in config.training_settings.keys():
        args[key] = getattr(config.training_settings, key)

    for key in config.misc.keys():
        args[key] = getattr(config.misc, key)

    args.save_dir = config.output.experiment_folder
    args.checkpoint_dir = config.output.checkpoint_dir

    if 'wandb' in args:
        args.wandb.dir = config.output.experiment_folder

    if args.get('resume', False) and args.get('pretrained_path', None) is not None:
        # Get the logs directory of the pretrained model
        experiment_directory = Path(args.pretrained_path).parent.parent

        if 'wandb' in args:
            # Pretrained model logged in wandb
            # Find the previous training log file and copy it to the new experiments output folder
            log_file = experiment_directory / 'training.log'
            if os.path.exists(log_file):
                shutil.copy(log_file, Path(args.save_dir) / 'training.log')

            # Copy the best model weights so far
            path_model = Path(args.pretrained_path).parents[0] / 'Model_best.pth'
            if os.path.exists(path_model):
                shutil.copy(path_model, Path(args.checkpoint_dir) / 'Model_best.pth')
        else:
            # Pretrained model logged in tensorboard
            experiment_tboard_log_dir = experiment_directory.parent / 'logs' / experiment_directory.name

            # Find the previous tensorboard files and copy them to the new experiments output folder
            if os.path.isdir(experiment_tboard_log_dir):
                tb_files = glob.glob(os.path.join(experiment_tboard_log_dir, 'events.*'))
                for tb_file in tb_files:
                    shutil.copy(tb_file, Path(args.checkpoint_dir) / Path(tb_file).name)
    else:
        args.resume = False
        args.pretrained_path = None

    args.max_seq_length = args.data.max_seq_length

    return Trainer(args, train_loader, val_loader, model, noise_scheduler, optimizer, scheduler)


def set_seed(seed: int) -> None:
    # Set the random seeds for repeatability
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)


def write_model_structure_to_file(
        filepath: str, model, batch_size: int, seq_length: int, in_channels: int, image_size: Tuple[int, int]
) -> None:
    """
    Writes the model architecture to a text file.

    Args:
        filepath:        str, path to the output text file.
        model:           nn.Module instance, model to be used for training.
        batch_size:      int, batch size.
        seq_length:      int, sequence length.
        in_channels:     int, number of input channels.
        image_size:      (int, int), tile size in pixels (width, height).
    """

    # Redirect stdout to file
    original = sys.stdout
    sys.stdout = open(filepath, "w", encoding="utf-8")


    if isinstance(model, MODELS['SDT']):
        torchinfo.summary(model.cuda(), input_size=[
            (batch_size, seq_length, in_channels, *image_size),    # input (image time series)
            (batch_size,),                                         # timestamp (timestep of adding noise)
            (batch_size, seq_length),                               # batch_positions (date sequence of the observations
            (batch_size, seq_length, 3, *image_size)               # expressed in #days since the first observation)
        ], device='cuda', depth=5)

    else:
        torchinfo.summary(model.cuda(), input_size=(batch_size, seq_length, in_channels, *image_size), device='cuda')
    torch.cuda.empty_cache()
    print('\n\n')
    print(model)

    # Reset stdout
    sys.stdout = original


def without_keys(d: Dict | DictConfig, ignore_keys: List[str]) -> Dict | DictConfig:
    """
    Returns a copy of the dictionary `d` without the keys listed in `ignore_keys`.
    """

    d_trim = {k: v for k, v in d.items() if k not in ignore_keys}

    if isinstance(d, DictConfig):
        return OmegaConf.create(d_trim)
    return d_trim
