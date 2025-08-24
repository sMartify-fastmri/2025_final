import torch
import argparse
import shutil
import os, sys
from pathlib import Path
from typing import List


if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

from utils.data.mraugment.data_augment import DataAugmentor


def parse():
    parser = argparse.ArgumentParser(description="Train sMaRtIfy's PromptMR+ on FastMRI challenge Images",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- HARDWARE SETUP, VERBOSE, AND PATHS ------------------------------------------------------------------------------
    parser.add_argument('-n', '--net-name', type=Path, default='test_PromptMR++', help='Name of network; recommended to contain hparams to distinguish model directory')
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../Data/val/', help='Directory of validation data')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-c', '--use-checkpoint', type=bool, default=True, help='Checkpointing to reduce GPU VRAM usage')
    parser.add_argument('-f', '--use-val-for-final', type=bool, default=False, help='Usage of validation data for training in final submission')

    parser.add_argument('--data-path-gamyun', type=Path, default='../Data_mask/', help='Directory of image mask data')
    parser.add_argument('--save-each-epoch', type=bool, default=False, help='Whether to save all the model in each epoch')

    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    # =====================================================================================================================
    
    
    # --- TRAINING TACTICS ------------------------------------------------------------------------------------------------
    parser.add_argument('-e', '--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size | Warning: Batch not implemented')
    parser.add_argument('-o', '--optimizer', type=str, default='AdamW', help='Optimizer model type')
    parser.add_argument('-l', '--lr', type=float, default=2e-4, help='Baseline learning rate')
    parser.add_argument('-s', '--lr-scheduler', type=str, default='StepLR', help='Learning rate scheduler type')

    parser.add_argument('--scheduler-start-epoch', type=int, default=5, help='Which epoch to start stepping scheduler')
    parser.add_argument('--regularizer-loss', type=str, default=None, help='Loss regularizer: None, L1, L2')
    parser.add_argument('--regularizer-lambda', type=float, default=1e-4, help='Lambda factor for regularizer')
    parser.add_argument('--regularizer-start-epoch', type=int, default=0, help='Which epoch to start applying regularizer')
    
    # Image mask configuration
    parser.add_argument('--gamyun-start-epoch', type=int, default=0, help='First epoch to apply mask to target image in train time')
    parser.add_argument('--gamyun-shade-strength', type=float, default=0., help='Value to fill the empty part of image mask. Should be under 1')
    parser.add_argument('--gamyun-loss', type=bool, default=True, help='Enable image mask area-based loss (alternative to slice-based loss)')
    parser.add_argument('--gamyun-loss-start-epoch', type=int, default=0, help='Which epoch to start enabling image mask area-based loss')

    # If --lr-scheduler == 'StepLR'
    parser.add_argument('--scheduler-step-size', type=int, default=11, help='First argument for StepLR')
    parser.add_argument('--scheduler-gamma', type=float, default=0.3, help='Second argument for StepLR')

    # Continuous planning
    parser.add_argument('--call-optimizer', type=lambda x: x.lower() == 'true', default=True, help='Call the optimizer of the previous run')
    parser.add_argument('--call-scheduler', type=lambda x: x.lower() == 'true', default=True, help='Call the scheduler of the previous run')
    
    # Calling another model to start
    parser.add_argument('--start-from-net', type=Path, default=None, help='Name of the saved network to start from')
    parser.add_argument('--start-from-best', type=lambda x: x.lower() == 'true', default=True, help='Whether to start from best model')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Epoch count of the saved network to start from')
    # =====================================================================================================================

    
    # --- MODEL HYPERPARAMETERS -------------------------------------------------------------------------------------------
    parser.add_argument('--num-cascades', type=int, default=8, help='Number of cascades | 12 in original PromptMR+')
    parser.add_argument('--n-feat0', type=int, default=8, help='Number of feat0 | 48 in original PromptMR+')
    parser.add_argument('--feature-dim', type=List[int], default=[24, 32, 40], help='Sizes of feature dimension | [72, 96, 120] in original PromptMR+')
    parser.add_argument('--prompt-dim', type=List[int], default=[8, 16, 24], help='Sizes of prompt dimension | [24, 48, 72] in original PromptMR+')
    parser.add_argument('--sens-n-feat0', type=int, default=8, help='Number of feat0 for sensitivity map | 24 in original PromptMR+')
    parser.add_argument('--sens-feature-dim', type=List[int], default=[12, 16, 20], help='Sizes of sensitivity feature dimension | [36, 48, 60] in original PromptMR+')
    parser.add_argument('--sens-prompt-dim', type=List[int], default=[4, 8, 12], help='Sizes of sensitivity-prompt dimension | [12, 24, 36] in original PromptMR+')
    parser.add_argument('--len-prompt', type=List[int], default=[3, 3, 3], help='Length of prompt vectors per level | [3, 3, 3] in original PromptMR+')
    parser.add_argument('--prompt-size', type=List[int], default=[16, 8, 4], help='Spatial size of prompt per level | [64, 32, 16] in original PromptMR+')
    parser.add_argument('--n-enc-cab', type=List[int], default=[2, 3, 3], help='Number of CABs in encoder blocks | [2, 3, 3] in original PromptMR+')
    parser.add_argument('--n-dec-cab', type=List[int], default=[2, 2, 3], help='Number of CABs in decoder blocks | [2, 2, 3] in original PromptMR+')
    parser.add_argument('--n-skip-cab', type=List[int], default=[1, 1, 1], help='Number of CABs in skip connections | [1, 1, 1] in original PromptMR+')
    parser.add_argument('--n-bottleneck-cab', type=int, default=3, help='Number of CABs in bottleneck block | 3 in original PromptMR+')
    parser.add_argument('--no-use-ca', type=bool, default=False, help='Disable channel attention module in CABs | False in original PromptMR+')

    # Plus implementations
    parser.add_argument('--adaptive-input', type=bool, default=True, help='Use residual adaptive input in Unet encoder | True in PromptMR+')
    parser.add_argument('--n-buffer', type=int, default=4, help='Number of internal feature buffers in PromptMRBlock | 4 in original PromptMR+')
    parser.add_argument('--n-history', type=int, default=3, help='Duration of history consideration | 11 in PromptMR+')
    parser.add_argument('--use-sens-adj', type=bool, default=True, help='Use adjacent slices when estimating sensitivity map | True in PromptMR+')
    # =====================================================================================================================


    # --- AUGMENTATION RULES ----------------------------------------------------------------------------------------------
    # Enable hacker
    parser.add_argument('--enable-hacker', type=lambda x: x.lower() == 'true', default=True, help='Hacker for the challenge, or helper')

    # If not hacker
    parser.add_argument('--mask-aug-weight', type=float, default=1., help='Kspace mask augmentation strength')
    parser.add_argument('--mask-aug-start', type=int, default=15, help='If not hacker, which epoch to start mask augmentation')
    parser.add_argument('--mask-aug-schedule', type=str, default='exp', help='Scheduling rule of kspace mask augmentation strength | Options: exp, const')
    parser.add_argument('--mask-aug-plateau-epoch', type=int, default=25, help='First epoch to apply constant mask augmentation strength')
    
    # MRAugment
    parser = DataAugmentor.add_augmentation_specific_args(parser)
    parser.add_argument('--annealing-epoch', type=int, default=1, help='Last epochs with constant augmentation strength; should be smaller than total epoch - aug_delay')
    # =====================================================================================================================
    
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    args.data_path_gamyun.mkdir(parents=True, exist_ok=True)

    train(args)
