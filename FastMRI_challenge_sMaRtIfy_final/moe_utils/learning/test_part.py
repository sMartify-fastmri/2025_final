import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from moe_utils.data.load_data_withclass import create_data_loaders_withclass
from moe_utils.model.moe_wrapper import MixtureOfPromptMR
from moe_utils.learning.classifier_part import call_classifier
from moe_utils.learning.moe_part import call_each_model

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _, _, class_label) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            class_label = class_label.cuda(non_blocking=True)
            output = model(kspace, mask, class_label)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None

def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    classifier = call_classifier(args)
    classifier.to(device=device)
    classifier.eval()

    model = MixtureOfPromptMR(args, checkpoints=None)
    all_checkpoints = call_each_model(args)
    model.load_each_state_dict(*all_checkpoints)
    model.to(device=device)

    forward_loader = create_data_loaders_withclass(data_path=args.data_path, args=args, classifier=classifier, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

    

    
    