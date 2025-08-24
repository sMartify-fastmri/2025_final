import math
import random
from typing import Optional, Callable

import torch
import numpy as np

def extract_acc(mask: np.array):
    width = mask.shape[0]
    total = np.sum(mask)

    acc = 4 if total / width > 0.24 else 8
    return acc

class MaskAugmentor():

    def __init__(
        self, 
        seed: int,
        aug_weight: float, 
        aug_start: int,
        aug_schedule: str,
        aug_plateau_epoch: int,
        current_epoch_fn: Callable[[], int] = None
    ):
        self.aug_weight = aug_weight
        self.aug_start = aug_start
        self.aug_schedule = aug_schedule
        self.aug_plateau_epoch = aug_plateau_epoch
        self.current_epoch_fn = current_epoch_fn or (lambda: 0)

        self.rng = np.random.RandomState(seed + 1000 if seed is not None else None)

    def schedule_p(self):
        if self.aug_schedule == 'exp':
            D = self.aug_start
            T = self.aug_plateau_epoch
            t = self.current_epoch_fn()
            p_max = self.aug_weight
            c = 5./(T-D)
            if t < D:
                return 0.0
            elif t >= T:
                return p_max
            else:
                return p_max/(1-math.exp(-(T-D)*c))*(1-math.exp(-(t-D)*c))
        elif self.aug_schedule == 'const':
            return self.aug_weight
        else:
            raise ValueError(f"Invalid schedule: {self.aug_schedule}")
        
    def __call__(
        self, 
        mask: np.array, 
        hacking: bool = False, 
        fix_acc: bool = False,
        fix_acc_as: Optional[int] = None
    ) -> np.array:
        if self.rng.uniform(0, 1) > self.schedule_p() and not hacking:
            return mask
            
        width = mask.shape[0]
                
        acc = (fix_acc_as or extract_acc(mask)) if fix_acc else self.rng.choice([4, 8])

        acs_length = round(0.08 * width) - 1
        acs_start = (width - acs_length) // 2
        acs_end = acs_start + acs_length

        mask_new = np.zeros_like(mask)
        mask_new[acs_start : acs_end + 1] = 1
        
        # Randomly distributed mask
        if self.rng.uniform(0, 1) < 0.5 and not hacking:
            # Randomly select 1/acc of the mask
            selection_mask = self.rng.rand(*mask.shape)
            mask_new[selection_mask < 1 / acc] = 1
        # Equispaced mask
        else:
            #start_idx = (width // 2) % acc if hacking else self.rng.choice(range(acc))
            # -> Force helper not to care about hacker region
            start_idx = (width // 2)
            if not hacking:
                while start_idx == (width // 2):
                    start_idx = self.rng.choice(range(acc))
            mask_new[start_idx :: acc] = 1

        return mask_new

        

        