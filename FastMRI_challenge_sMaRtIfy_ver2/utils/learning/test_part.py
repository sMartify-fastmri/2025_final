import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.promptmr_plus import PromptMR

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _, _) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

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

    model = PromptMR(
        num_cascades=args.num_cascades,
        num_adj_slices=1,   
        n_feat0=args.n_feat0,
        feature_dim=args.feature_dim,
        prompt_dim=args.prompt_dim,
        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim=args.sens_feature_dim,
        sens_prompt_dim=args.sens_prompt_dim,
        len_prompt=args.len_prompt,
        prompt_size=args.prompt_size,
        n_enc_cab=args.n_enc_cab,
        n_dec_cab=args.n_dec_cab,
        n_skip_cab=args.n_skip_cab,
        n_bottleneck_cab=args.n_bottleneck_cab,
        no_use_ca=args.no_use_ca,
        adaptive_input=args.adaptive_input,
        n_buffer=args.n_buffer,
        n_history=args.n_history,
        use_sens_adj=args.use_sens_adj
    )
    model.to(device=device)
    
    if args.use_best:
        checkpoints = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    else:
        checkpoints = torch.load(args.exp_dir / f'model_epoch{args.use_epoch}in{args.model_last_epoch}.pt', map_location='cpu', weights_only=False)
    print(checkpoints['epoch'], checkpoints['best_val_loss'].item())
    model.load_state_dict(checkpoints['model'])
    
    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)