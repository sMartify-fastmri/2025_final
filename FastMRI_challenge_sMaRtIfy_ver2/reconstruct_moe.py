import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from moe_utils.learning.test_part import forward
import time

from typing import List

def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--classifier-name', type=str, default='awesome_classifier_ver2')
    parser.add_argument('--classifier-path', type=Path, default='../result/classifier/')
    
    parser.add_argument('--call-moe-units', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--starting-expert', type=Path, default='test_PromptMR++_hacker_baseline')
    parser.add_argument('--starting-epoch', type=int, default=30)
    parser.add_argument('--starting-expert-max-epoch', type=int, default=45)
    parser.add_argument('--call-moe-name', type=Path, default='test_PromptMR++_hacker_moe_clutch')
    parser.add_argument('--max-epoch2call', type=int, default=35)
    parser.add_argument('--brain-acc4-epoch2call', type=int, default=35)
    parser.add_argument('--brain-acc8-epoch2call', type=int, default=35)
    parser.add_argument('--knee-acc4-epoch2call', type=int, default=35)
    parser.add_argument('--knee-acc8-epoch2call', type=int, default=35)
    
    
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_PromptMR++_hacker_moe_clutch', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='../Data/leaderboard/', help='Directory of test data')
    
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')

    parser.add_argument('--use-best', type=lambda x: x.lower() == 'true', default=False, help='Call the best validated model')
    parser.add_argument('--use-epoch', type=int, default=0, help='Model epoch to use')
    parser.add_argument('--model-last-epoch', type=int, default=50, help='Figures out the file name')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--mask-aug-weight', type=float, default=1., help='Kspace mask augmentation strength')
    parser.add_argument('--mask-aug-start', type=int, default=15, help='If not hacker, which epoch to start mask augmentation')
    parser.add_argument('--mask-aug-schedule', type=str, default='exp', help='Scheduling rule of kspace mask augmentation strength | Options: exp, const')
    parser.add_argument('--mask-aug-plateau-epoch', type=int, default=25, help='First epoch to apply constant mask augmentation strength')

    # --- MODEL HYPERPARAMETERS -----------------------------------------------------------------------
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
    # =======================================================================================

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'

    start_time = time.time()
    
    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')