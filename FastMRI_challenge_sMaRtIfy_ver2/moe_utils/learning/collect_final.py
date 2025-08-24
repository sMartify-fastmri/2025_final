
import numpy as np
import torch

from utils.common.utils import save_reconstructions
from moe_utils.data.load_data_withclass import create_data_loaders_withclass
from moe_utils.learning.classifier_part import call_classifier
from moe_utils.learning.test_part import test
from moe_utils.model.broker import HackingSystem


def call_single_checkpoint(full_path):
    checkpoints = torch.load(full_path, map_location='cpu', weights_only=False)
    return checkpoints

def save_full_checkpoints(args, model):
    model_name = 'best_model.pt'
    torch.save({'model': model.state_dict()}, f=args.exp_dir / model_name)
    print(f'Final model saved: {model_name} in {str(args.exp_dir)}')

def forward(args):
    
    # Load checkpoints separately
    print('Loading checkpoints separately...')
    hacker_checkpoints, helper_checkpoints = [], []

    hacker_checkpoints.append(call_single_checkpoint(args.path_hacker_brain_acc4))
    hacker_checkpoints.append(call_single_checkpoint(args.path_hacker_brain_acc8))
    hacker_checkpoints.append(call_single_checkpoint(args.path_hacker_knee_acc4))
    hacker_checkpoints.append(call_single_checkpoint(args.path_hacker_knee_acc8))

    helper_checkpoints.append(call_single_checkpoint(args.path_helper_brain_acc4))
    helper_checkpoints.append(call_single_checkpoint(args.path_helper_brain_acc8))
    helper_checkpoints.append(call_single_checkpoint(args.path_helper_knee_acc4))
    helper_checkpoints.append(call_single_checkpoint(args.path_helper_knee_acc8))

    model = HackingSystem(args, hacker_checkpoints, helper_checkpoints)
    save_full_checkpoints(args, model)
    print('Saved checkpoints as a single model...')
