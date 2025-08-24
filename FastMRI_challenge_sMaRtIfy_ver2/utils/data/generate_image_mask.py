
import numpy as np
import h5py
import cv2
from tqdm import tqdm

from pathlib import Path

def generate(args, train_dir, val_dir):
    train_files = list((Path(train_dir) / 'image').iterdir())
    val_files = list((Path(val_dir) / 'image').iterdir())

    for fname in tqdm(sorted(train_files), desc='Generating image masks from train files'):
        generate_mask_fromfile(fname=fname, mask_dir=args.data_path_gamyun)
    for fname in tqdm(sorted(val_files), desc='Generating image masks from val files'):
        generate_mask_fromfile(fname=fname, mask_dir=args.data_path_gamyun)

def generate_mask_fromfile(fname, mask_dir: Path):
    is_brain = (str(fname).split('/')[-1].split('_')[-3] == 'brain')
    cutoff = 5e-5 if is_brain else 2e-5

    with h5py.File(fname, 'r') as hf:
        num_slices = hf['image_label'].shape[0]
    for i_slice in range(num_slices):
        mask_fname = mask_dir / f"{(str(fname).split('/')[-1]).replace('.h5', '')}_s{i_slice}.npy"
        if mask_fname.exists():
            continue
            
        with h5py.File(fname, 'r') as hf:       
            target = hf['image_label'][i_slice]
            mask = generate_mask(target, cutoff)
            np.save(mask_fname, mask)
    

def generate_mask(target, cutoff):
    mask = np.zeros(target.shape)
    mask[target > cutoff] = 1

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)

    return mask
    