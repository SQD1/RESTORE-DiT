import argparse
import logging
import logging.config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from lib import config_utils, data_utils, utils
from lib.formatter import RawFormatter
from lib.logger import prepare_logger
from diffusers.schedulers import DDIMScheduler

parser = ArgumentParser(
    description='RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer (Training)',
    formatter_class=RawFormatter
)
parser.add_argument(
    'config_file', type=str, help='yaml configuration file to augment/overwrite the settings in configs/default.yaml'
)
parser.add_argument(
    '--save_dir', type=str, required=True, help='Path to the directory where models and logs should be saved'
)
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb_project', type=str, default='utilise', help='Wandb project name')


def main(args: argparse.Namespace) -> None:
    prog_name = 'RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer (Training)'
    print('\n{}\n{}\n'.format(prog_name, '=' * len(prog_name)))

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f'ERROR: Cannot find the yaml configuration file: {args.config_file}')

    # Import the user configuration file
    cfg_custom = config_utils.read_config(args.config_file)

    if not cfg_custom:
        sys.exit(1)

    # Augment/overwrite the default parameter settings with the runtime arguments given by the user
    cfg_default = config_utils.read_config('configs/default_train.yaml')
    config = OmegaConf.merge(cfg_default, cfg_custom)
    config.output.output_directory = args.save_dir

    if args.wandb:
        config.wandb = OmegaConf.create()
        config.wandb.project = args.wandb_project

    # Create the output directory. The name of the output directory is a combination of the current date, time, and an
    # optional suffix.
    config.output.experiment_folder = utils.create_output_directory(config)

    # Set up the logger
    log_file = os.path.join(config.output.experiment_folder, 'run.log') if config.output.experiment_folder else None
    logger = prepare_logger('root_logger', level=logging.INFO, log_to_console=True, log_file=log_file)

    # Print runtime arguments to the console
    logger.info('Configuration file: %s', args.config_file)
    logger.info('\nSettings\n--------\n')
    config_utils.print_config(config, logger=logger)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # ------------------------------------------------- Data loaders ------------------------------------------------- #
    logger.info('\nInitialize data loader (training set)...')
    train_loader = data_utils.get_dataloader(
        config, phase='train', pin_memory=config.misc.pin_memory, drop_last=True, logger=logger
    )
    logger.info('Initialize data loader (validation set)...\n')
    val_loader = None

    logger.info('Number of training samples: %d', train_loader.dataset.__len__())

    # ----------------------------------------- Prepare the output directory ----------------------------------------- #
    logger.info('\nPrepare output folders and files\n--------------------------------\n')

    # Save the path of the checkpoint directory
    config.output.checkpoint_dir = os.path.join(config.output.experiment_folder, 'checkpoints')
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    logger.info('Model weights will be stored in: %s\n', config.output.checkpoint_dir)

    # Write the runtime configuration to file
    config_file = os.path.join(config.output.experiment_folder, 'config.yaml')
    config_utils.write_config(config, config_file)

    # ----------------------------------------------- Define the model ----------------------------------------------- #
    logger.info('\nModel Architecture\n------------------\n')
    logger.info('Architecture: %s', config.method.model_type)


    # input_dim = train_loader.dataset.num_channels
    input_dim = 10   # 4 for RGB_NIR

    model, args_model = utils.get_model(config, input_dim, logger)

    logger.info('Number of trainable parameters: %d\n', utils.count_model_parameters(model))

    # Log model parameters to file
    config_file = os.path.join(config.output.experiment_folder, 'model_config.yaml')
    config_utils.write_config(OmegaConf.create({config.method.model_type: args_model}), config_file)

    # Write model architecture to txt file
    if config.output.plot_model_txt:
        file = os.path.join(config.output.experiment_folder, 'model_parameters.txt')
        logger.info('Writing model architecture to file: %s\n', file)
        utils.write_model_structure_to_file(
            file, model, config.training_settings.batch_size, train_loader.dataset.max_seq_length, input_dim,
            train_loader.dataset.image_size
        )

    model = nn.DataParallel(model)
    # --------------------------------------------------- Training --------------------------------------------------- #
    logger.info('\nPrepare training\n----------------\n')
    logger.info('Python version: %s', sys.version)
    logger.info('Torch version: %s', torch.__version__)
    logger.info('CUDA version: %s\n', torch.version.cuda)

    # Get optimizer and learning rate scheduler
    optimizer = utils.get_optimizer(config, model, logger)
    scheduler = utils.get_scheduler(config, optimizer, train_loader.dataset.__len__(), logger)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # Initialize the trainer and start training
    trainer = utils.get_trainer(config, train_loader, val_loader, model, noise_scheduler, optimizer, scheduler)
    trainer.train()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    main(parser.parse_args())
