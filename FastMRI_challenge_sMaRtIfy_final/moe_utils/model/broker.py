
import torch
import torch.nn as nn

from moe_utils.model.moe_wrapper import MixtureOfPromptMR
from utils.data.mask_augment import MaskAugmentor


# TODO: device, dtype 검토
class Broker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.broker = MaskAugmentor(
            seed=args.seed,
            aug_weight=0,
            aug_start=0,
            aug_schedule='const',
            aug_plateau_epoch=0,
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.flatten() # TODO: check the shape of mask
        hacker_mask = self.broker(mask.cpu().numpy(), hacking=True, fix_acc=True)  # TODO: check the shape of mask
        hacker_mask = torch.from_numpy(hacker_mask).to(mask.device)
        
        if torch.equal(hacker_mask, mask):
            return torch.tensor([1, 0], device=mask.device) # Send to HACKER
        else: 
            for i in range(16):
                candidate_mask = hacker_mask[i:i+16]
                if torch.equal(candidate_mask, mask[:16]):
                    return torch.tensor([0, 1], device=mask.device) # Send to HELPER
            return torch.tensor([0, 1], device=mask.device) # Send to HELPER


class HackingSystem(nn.Module):
    def __init__(self, args, hacker_checkpoints, helper_checkpoints):

        super().__init__()

        self.broker = Broker(args)
        self.hacker = self.make_experts(args)
        self.helper = self.make_experts(args)

        if hacker_checkpoints is not None and helper_checkpoints is not None:
            self.load_state_dicts(hacker_checkpoints, helper_checkpoints)

        self.broker_friends = nn.ModuleList([self.hacker, self.helper])
        self.broker_count = [0, 0]

    def make_experts(self, args):
        return MixtureOfPromptMR(args, checkpoints=None)

    def load_state_dicts(self, hacker_checkpoints, helper_checkpoints):
        self.hacker.load_each_state_dict(*hacker_checkpoints)
        self.helper.load_each_state_dict(*helper_checkpoints)

    def tea_bagging(self):
        if self.broker_count[0] == 0  and self.broker_count[1] == 0:
            print("Before forwarding.")
        else:
            hacking_ratio = self.broker_count[0] / (self.broker_count[0] + self.broker_count[1])
            print(f'Hacking ratio: {hacking_ratio} over 1')
            if hacking_ratio > 0.7:
                print("<<~~~|| Your bias is our benchmark. HACKER says thanks :) ||~~~>>")
            elif hacking_ratio < 0.3:
                print("<<~~~|| Overfitting bait? We passed. HELPER handled it :) ||~~~>>")
            else:
                print("<<~~~|| Bluff and pattern. BROKER routed them all :) ||~~~>>")
        
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, class_label: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(self.broker(mask)).item()
        self.broker_count[idx] += 1
        return self.broker_friends[idx](masked_kspace, mask, class_label)
            

            

        
        
        
        