import logging
import os
import sys
from typing import Optional

import prodict
import yaml
from omegaconf import DictConfig, OmegaConf
from prodict import Prodict


class _PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


_PrettySafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _PrettySafeLoader.construct_python_tuple)


def resolve_tuple(*args):
    """Resolves a value in a config file with a structure like ${tuple:1,2} to a tuple (1, 2)."""
    return tuple(args)


OmegaConf.register_new_resolver('tuple', resolve_tuple)


def print_config(config: DictConfig | prodict.Prodict | str, logger: Optional[logging.Logger] = None) -> None:
    """
    Prints a yaml configuration file to the console.

    Args:
        config:      string or dict, path of the yaml file or previously imported configuration file.
        logger:      logger instance.
    """

    if isinstance(config, Prodict):
        config = config.to_dict(is_recursive=True)
    elif isinstance(config, str):
        config = read_config(config)
    elif isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)

    if logger:
        logger.info(yaml.dump(config, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True))
    else:
        yaml.dump(config, sys.stdout, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True)


def read_config(file: str) -> DictConfig:
    """
    Reads a yaml configuration file.

    Args:
        file:  str, path of the yaml configuration file.

    Returns:
        dict (DictConfig), imported yaml file.
    """

    # Load configuration from file
    if not os.path.exists(file):
        raise FileNotFoundError(f'ERROR: Cannot find the file {file}\n')
    try:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=_PrettySafeLoader)
    except yaml.YAMLError as e:
        raise RuntimeError(f'ERROR: Cannot load the file {file}\n') from e

    return OmegaConf.create(config)


def write_config(data: DictConfig | prodict.Prodict, outfile: str) -> None:
    """
    Writes the dictionary data to a yaml file.

    Args:
        data:     dict (DictConfig), data to be stored as a yaml file.
        outfile:  str, path of the output file.
    """

    with open(outfile, "w", encoding="utf-8") as f:
        if isinstance(data, Prodict):
            yaml.dump(data.to_dict(is_recursive=True), f, indent=4, default_flow_style=None, sort_keys=False,
                      allow_unicode=True)
        elif isinstance(data, DictConfig):
            OmegaConf.save(config=data, f=f)
        else:
            yaml.dump(data, f, indent=4, default_flow_style=None, sort_keys=False, allow_unicode=True)
