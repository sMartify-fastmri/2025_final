import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch

class ClassifierDataset(Dataset):
    def __init__(self, root, forward=False):
        self.forward = forward
        self.image_examples = []

        image_files = list(Path(root / "image").iterdir())
        for fname in sorted(image_files):
            num_slices = self._get_metadata(fname)
            
            self.image_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
    
    def _get_metadata(self, fname):
        num_slices = 0
        with h5py.File(fname, "r") as hf:
            if 'image_grappa' in hf.keys():
                num_slices = hf['image_grappa'].shape[0]
            return num_slices

    def __len__(self):
        return len(self.image_examples)

    def __getitem__(self, i):
        image_fname, dataslice = self.image_examples[i]
        with h5py.File(image_fname, "r") as hf:
            image_grappa = hf['image_grappa'][dataslice]

        if self.forward:
            target = -1
        else:
            target = torch.tensor([1, 0]) if image_fname.name.split('_')[0] == 'brain' else torch.tensor([0, 1])

        return image_grappa, target.float()

def create_classifier_data_loader(data_path, args, shuffle=False, isforward=False):
    return DataLoader(
        ClassifierDataset(root=data_path, forward=isforward),
        batch_size=args.batch_size,
        shuffle=shuffle
    )