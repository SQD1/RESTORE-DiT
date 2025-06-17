import argparse
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from prodict import Prodict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lib import config_utils
from lib.arguments import eval_parser
from lib.data_utils import get_dataset
# from lib.eval_tools import Imputation
from lib.eval_tools_diffusion import Imputation
from lib.logger import AverageMeter
from lib.metrics import EvalMetrics
from torchvision import transforms

trans_scale = transforms.Normalize([0.5], [0.5])
generator = torch.manual_seed(19)

def print_stats(stats, evaluator, print_only_masked=False):
    prefix = evaluator.compute_metrics.prefix

    if print_only_masked is False:
        print('Metrics computed over all pixels:')
        for k, v in stats.items():
            if 'occluded_input_pixels' in k or 'observed_input_pixels' in k:
                pass
            else:
                metric = k.replace(prefix, '')
                print(f'{metric.upper()}: {v}')

    if evaluator.compute_metrics.eval_occluded_observed:
        print('\nMetrics computed over all masked input pixels:')
        for k, v in stats.items():
            if 'occluded_input_pixels' in k:
                metric = k.replace(prefix, '').replace('_occluded_input_pixels', '').replace('_images', '')
                print(f'{metric.upper()}: {v}')

        if print_only_masked is False:
            print('\nMetrics computed over all observed input pixels:')
            for k, v in stats.items():
                if 'observed_input_pixels' in k:
                    metric = k.replace(prefix, '').replace('_observed_input_pixels', '').replace('_images', '')
                    print(f'{metric.upper()}: {v}')

def save_stats_to_file(stats, evaluator, file_path, print_only_masked=False):
    prefix = evaluator.compute_metrics.prefix

    with open(file_path, 'w') as f:
        if print_only_masked is False:
            f.write('Metrics computed over all pixels:\n')
            for k, v in stats.items():
                if 'occluded_input_pixels' in k or 'observed_input_pixels' in k:
                    pass
                else:
                    metric = k.replace(prefix, '')
                    f.write(f'{metric.upper()}: {v}\n')

        if evaluator.compute_metrics.eval_occluded_observed:
            f.write('\nMetrics computed over all masked input pixels:\n')
            for k, v in stats.items():
                if 'occluded_input_pixels' in k:
                    metric = k.replace(prefix, '').replace('_occluded_input_pixels', '').replace('_images', '')
                    f.write(f'{metric.upper()}: {v}\n')

            if print_only_masked is False:
                f.write('\nMetrics computed over all observed input pixels:\n')
                for k, v in stats.items():
                    if 'observed_input_pixels' in k:
                        metric = k.replace(prefix, '').replace('_observed_input_pixels', '').replace('_images', '')
                        f.write(f'{metric.upper()}: {v}\n')

class Evaluator:
    def __init__(self, args: argparse.Namespace, args_test_data: DictConfig):
        self.args = args

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args_metrics = {
            'masked_metrics': True,  # True： use real cloud mask to maskout the cloud pixels in targets
            'sam_units': 'deg',
            'eval_occluded_observed': True,
            'mae': True, 'rmse': True, 'mse': False, 'ssim': True, 'psnr': True, 'sam': True
        }

        self.compute_metrics = EvalMetrics(self.args_metrics)
        _ = torch.set_grad_enabled(False)

        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f'Cannot find the configuration file used during training: {args.config_file}\n')

        # Read config file used during training
        self.config = config_utils.read_config(args.config_file)

        # Merge generic data settings (used during training) with test-specific data settings
        self.config.data.update(args_test_data)
        self.config.data.preprocessed = True

        # Evaluate the entire image sequence
        self.config.data.max_seq_length = None

        # Get the data loader
        dset = get_dataset(self.config, phase='test')
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dset, batch_size=1, shuffle=False, num_workers=self.config.misc.num_workers, drop_last=False
        )

        # Get the imputation model
        self.imputation = Imputation(
            config_file_train=self.args.config_file,
            method=self.args.method,
            mode=args.mode,
            checkpoint=self.args.checkpoint,
            multigpus=True,
            num_inference_steps=args.inference_steps,
            ifDate=True,    # False or True
            ifCond=True,
            generator=generator
        )

    def evaluate(self):
        self._initialize_stats()

        for i, batch in enumerate(tqdm(self.dataloader, leave=False)):
            if torch.any(batch['masks']):

                # rescale [-1, 1]
                if self.args.method == 'SDT':
                    batch['x'] = trans_scale(batch['x'])

                _, y_pred = self.imputation.impute_sample(batch)

                # Evaluation
                metrics = self.compute_metrics(batch, y_pred)

                for key, value in metrics.items():
                    self.stats[key].update(value)

        # Average metrics over all samples
        for metric in self.stats.keys():
            self.stats[metric] = self.stats[metric].avg

        return self.stats

    def _initialize_stats(self):
        stats = Prodict()
        eval_occluded_observed = self.args_metrics.get('eval_occluded_observed', True)

        for metric, val in self.args_metrics.items():
            if metric in ['masked_metrics', 'sam_units', 'eval_occluded_observed']:
                pass
            elif val:
                metric_name = f'masked_{metric}' if (
                        self.args_metrics['masked_metrics'] and 'ssim' not in metric
                ) else metric
                stats[metric_name] = AverageMeter()

                if eval_occluded_observed and 'ssim' not in metric:
                    stats[f'{metric_name}_occluded_input_pixels'] = AverageMeter()
                    stats[f'{metric_name}_observed_input_pixels'] = AverageMeter()

                if eval_occluded_observed and 'ssim' in metric:
                    stats[f'{metric_name}_images_occluded_input_pixels'] = AverageMeter()
                    stats[f'{metric_name}_images_observed_input_pixels'] = AverageMeter()

        self.stats = stats


if __name__ == '__main__':

    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    # Extract settings w.r.t. test data
    if args.test_data.test_config is not None:
        if not os.path.isfile(args.test_data.test_config):
            raise FileNotFoundError(f'Cannot find the test configuration file: {args.test_data.test_config}\n')
        args_test_data = config_utils.read_config(args.test_data.test_config).data
    else:
        args_test_data = OmegaConf.create()

        if args.test_data.data_dir is not None:
            if not os.path.exists(args.test_data.data_dir):
                raise ValueError(f'Cannot find the data directory: {args.test_data.data_dir}\n')
            args_test_data.root = args.test_data.data_dir
        if args.test_data.hdf5_file is not None:
            if not os.path.isfile(os.path.join(args_test_data.root, args.test_data.hdf5_file)):
                raise FileNotFoundError(
                    f'Cannot find the data file: {os.path.join(args_test_data.root, args.test_data.hdf5_file)}\n')
            args_test_data.hdf5_file = args.test_data.hdf5_file
        if args.test_data.split is not None:
            args_test_data.split = args.test_data.split
        if args.test_data.mode is not None:
            args_test_data.mode = args.test_data.mode

        args_test_data.rescale = args.test_data.rescale

    evaluator = Evaluator(args, args_test_data)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print('Evaluation completed in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    print('Statistics:\n===========')
    # for k, v in stats.items():
    #     print(f'{k}: {v}')
    print_stats(stats, evaluator)
