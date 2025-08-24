
import torch
import torch.nn as nn

from utils.model.promptmr_plus import PromptMR


class MixtureOfPromptMR(nn.Module):
    def __init__(self, args, checkpoints):
        super().__init__()

        self.brain_acc4 = self.make_experts(args)
        self.brain_acc8 = self.make_experts(args)
        self.knee_acc4 = self.make_experts(args)
        self.knee_acc8 = self.make_experts(args)
        
        if checkpoints is not None:
            self.load_state_dicts(checkpoints)
        self.experts = nn.ModuleList([self.brain_acc4, self.brain_acc8, self.knee_acc4, self.knee_acc8])


    def make_experts(self, args):
        return PromptMR(
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

    def load_state_dicts(self, checkpoints):
        self.brain_acc4.load_state_dict(checkpoints)
        self.brain_acc8.load_state_dict(checkpoints)
        self.knee_acc4.load_state_dict(checkpoints)
        self.knee_acc8.load_state_dict(checkpoints)

    def load_each_state_dict(self, checkpoints_brain_acc4, checkpoints_brain_acc8, checkpoints_knee_acc4, checkpoints_knee_acc8):
        self.brain_acc4.load_state_dict(checkpoints_brain_acc4['model'])
        self.brain_acc8.load_state_dict(checkpoints_brain_acc8['model'])
        self.knee_acc4.load_state_dict(checkpoints_knee_acc4['model'])
        self.knee_acc8.load_state_dict(checkpoints_knee_acc8['model'])

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, class_label: torch.Tensor, use_checkpoint: bool = True):
        idx = class_label.argmax().item()

        return self.experts[idx](masked_kspace, mask, use_checkpoint=use_checkpoint)