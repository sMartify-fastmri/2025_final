import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch


def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with a unique but deterministic seed.
    This ensures reproducible results across different runs while maintaining
    diversity between workers.
    """
    # Get the initial seed from the main process
    worker_seed = torch.initial_seed() % 2**32
    # Make each worker have a different but deterministic seed
    worker_seed = worker_seed + worker_id
    
    # Set seeds for this worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        self.gamyun_path = self.transform.gamyun_path

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            self.max_gamyun_areas = {}
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

                if self.gamyun_path is not None:
                    max_gamyun_area = 0.
                    for i_slice in range(num_slices):
                        gamyun_fname = self.gamyun_path / f"{fname.name.replace('.h5', '')}_s{i_slice}.npy"
                        if gamyun_fname.exists():
                            gamyun = np.load(gamyun_fname)
                            max_gamyun_area = max(max_gamyun_area, np.sum(gamyun))
                    self.max_gamyun_areas[fname.name] = max_gamyun_area


        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
            max_gamyun_area = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            max_gamyun_area = self.max_gamyun_areas[image_fname.name]
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, max_gamyun_area)




def create_data_loaders(data_path, args, shuffle=False, isforward=False, augmentor=None):
    if isforward == False: # Train: not isforward, but shuffle
        max_key_ = args.max_key
        target_key_ = args.target_key
        gamyun_path = args.data_path_gamyun
        enable_hacker = args.enable_hacker
        if shuffle == False:
            augmentor = None # Validation: not isforward, not shuffle
    else: # Evaluation: isforward, not shuffle
        max_key_ = -1
        target_key_ = -1
        gamyun_path = None
        enable_hacker = False
        augmentor = None
    
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            args,
            isforward, 
            max_key_, 
            gamyun_path=gamyun_path,
            augmentor=augmentor, 
            hacker=enable_hacker          
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    # Use single-process loading to avoid CUDA initialization issues in workers
    # Multi-process loading with CUDA operations in transforms causes fork() issues
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,  # Single-process to avoid CUDA fork issues
        worker_init_fn=worker_init_fn
    )
    return data_loader


