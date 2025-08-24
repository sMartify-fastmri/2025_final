import numpy as np
import torch

from pathlib import Path

from moe_utils.data.mask_augment import MaskAugmentor, extract_acc
from utils.data.generate_image_mask import generate_mask

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransformWithClass:
    def __init__(self, args, isforward, max_key, gamyun_path=None, brain_augmentor=None, knee_augmentor=None, hacker=False):
        self.isforward = isforward
        self.max_key = max_key
        self.gamyun_path = gamyun_path
        self.brain_augmentor = brain_augmentor
        self.knee_augmentor = knee_augmentor
        self.use_augment = (brain_augmentor is not None) or (knee_augmentor is not None)
        self.hacker = hacker
        self.mask_augmentor = MaskAugmentor(
            seed=args.seed,
            aug_weight=args.mask_aug_weight,
            aug_start=args.mask_aug_start,
            aug_schedule=args.mask_aug_schedule,
            aug_plateau_epoch=args.mask_aug_plateau_epoch,
            current_epoch_fn=(self.brain_augmentor.current_epoch_fn if self.use_augment else None)
        )

        self.dummy_classifier = (str(args.classifier_name) == 'dummy')

    def __call__(self, mask, input, target, attrs, fname, slice, max_gamyun_area, is_brain):
        # Forced classifier for determined file name
        if self.dummy_classifier and not self.isforward:
            is_brain = (str(fname).split('/')[-1].split('_')[-3] == 'brain')

        # Classifier checking
        if (not self.isforward) and (not (is_brain == (str(fname).split('/')[-1].split('_')[-3] == 'brain'))):
            print(f"classifier exception! {str(fname).split('/')[-1]}")

        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            gamyun = np.load(self.gamyun_path / f"{str(fname).replace('.h5', '')}_s{slice}.npy")
            max_gamyun_ratio = np.sum(gamyun) / max_gamyun_area if max_gamyun_area > 0 else 1.
            gamyun = to_tensor(gamyun)
        else:
            target = -1
            maximum = -1
            gamyun = -1    
            max_gamyun_ratio = -1
        
        # Apply mask augmentation
        if self.use_augment:
            if self.hacker:
                mask = self.mask_augmentor(mask=mask, hacking=True)
            else:
                mask = self.mask_augmentor(mask=mask, hacking=False)

        # Apply MRAugment if available and configured
        if self.use_augment and not self.isforward and target is not None:
            # Convert kspace to the format expected by MRAugment (complex tensor with real/imag as last dim)
            if input.dtype.is_complex:
                kspace_for_aug = torch.stack([input.real, input.imag], dim=-1)
            else:
                kspace_for_aug = input

            augmentor = self.brain_augmentor if is_brain else self.knee_augmentor
            
            # Get current augmentation probability
            if hasattr(augmentor, 'schedule_p') and augmentor.schedule_p() > 0.0:
                # Apply augmentation
                augmented_input, augmented_target = augmentor(kspace_for_aug, target.shape)
                
                # Convert back to the format expected by the model
                if augmented_input.shape[-1] == 2:  # real/imag format
                    input = augmented_input[..., 0] + 1j * augmented_input[..., 1]
                    target = augmented_target

                    # Re-generating image mask
                    is_brain = (str(fname).split('/')[-1].split('_')[-3] == 'brain')
                    cutoff = 5e-5 if is_brain else 2e-5
                    gamyun = generate_mask(target.cpu(), cutoff)
                    gamyun = to_tensor(gamyun)

        # Class label
        acc = extract_acc(mask)
        if is_brain:
            if acc == 4:
                class_label = torch.tensor([1, 0, 0, 0]) # brain_acc4
            else:
                class_label = torch.tensor([0, 1, 0, 0]) # brain_acc8   
        else:
            if acc == 4:
                class_label = torch.tensor([0, 0, 1, 0]) # knee_acc4
            else:
                class_label = torch.tensor([0, 0, 0, 1]) # knee_acc8
                    
        kspace = input * mask
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()


        return mask, kspace, target, maximum, fname, slice, gamyun, max_gamyun_ratio, class_label
