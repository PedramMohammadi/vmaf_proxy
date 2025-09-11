#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random
import gcsfs
import re

class VMAFDataset(Dataset):

    def __init__(self, csv_file, root_dir, crop_size=128, fixed_crop=False, 
                 n_frames=3, split=None):
        
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.fs = gcsfs.GCSFileSystem()
        
        if split:
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.fixed_crop = fixed_crop
        self.n_frames = n_frames
        self.to_tensor = transforms.ToTensor()

        if n_frames % 2 == 0:
            raise ValueError(f"--n_frames must be odd (got {n_frames})")

        # Collect frame keys
        all_dist_keys = self._sorted_keys(self.data.columns, "dist_frame_")
        all_ref_keys  = self._sorted_keys(self.data.columns, "ref_frame_")

        if len(all_dist_keys) != len(all_ref_keys):
            raise ValueError("Mismatched number of distorted vs reference frame columns.")
        if len(all_dist_keys) < n_frames:
            raise ValueError(f"Need at least {n_frames} frame columns (got {len(all_dist_keys)}).")

        center = len(all_dist_keys) // 2
        half = n_frames // 2
        start = max(0, min(center - half, len(all_dist_keys) - n_frames))
        self.dist_keys = all_dist_keys[start:start+n_frames]
        self.ref_keys  = all_ref_keys[start:start+n_frames]

    def _sorted_keys(self, cols, prefix):
        keys = [c for c in cols if c.startswith(prefix)]
        if not keys:
            raise ValueError(f"No columns starting with {prefix}")
        keys.sort(key=lambda s: int(re.search(r'(\d+)$', s).group(1)))
        return keys

    def __len__(self):
        return len(self.data)

    def load_and_crop(self, relative_path, i, j):
        t = self._open_image_as_tensor(relative_path)      # CHW in [0,1], float32

        C, H, W = t.shape
        # Clamp AFTER padding so indices are always valid
        i = 0 if H <= self.crop_size else min(i, H - self.crop_size)
        j = 0 if W <= self.crop_size else min(j, W - self.crop_size)

        patch = t[:, i:i + self.crop_size, j:j + self.crop_size]
        return patch.contiguous()
    
    def _open_image_as_tensor(self, relative_path: str):
        # building full path without introducing backslashes on GCS
        if str(self.root_dir).startswith("gs://"):
            full_path = f"{str(self.root_dir).rstrip('/')}/{str(relative_path).lstrip('/')}"
            with self.fs.open(full_path, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('L')
                    img = img.copy()
        else:
            full_path = os.path.join(self.root_dir, os.path.normpath(relative_path))
            with open(full_path, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('L')
                    img = img.copy()
        return self.to_tensor(img)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- open first distorted frame ONCE ---
        first_rel = row[self.dist_keys[0]]
        t0 = self._open_image_as_tensor(first_rel)

        # --- choose a single crop (shared by all frames) ---
        H, W = t0.shape[1], t0.shape[2]
        if self.fixed_crop:
            i = max(0, (H - self.crop_size) // 2)
            j = max(0, (W - self.crop_size) // 2)
        else:
            i = 0 if H <= self.crop_size else random.randint(0, H - self.crop_size)
            j = 0 if W <= self.crop_size else random.randint(0, W - self.crop_size)

        # --- distorted stack: reusing the first frame tensor; load the rest normally ---
        patch0 = t0[:, i:i + self.crop_size, j:j + self.crop_size].contiguous()
        dist_rest = [self.load_and_crop(row[k], i, j) for k in self.dist_keys[1:]]
        x_dist = torch.cat([patch0] + dist_rest, dim=0)   # (T, Hc, Wc) with T = n_frames

        # --- reference stack: same crop corrdinates, load each once ---
        x_ref = torch.cat([self.load_and_crop(row[k], i, j) for k in self.ref_keys], dim=0)

        # --- target ---
        vmaf = float(row["vmaf_score"])
        y = torch.tensor(vmaf / 100.0, dtype=torch.float32)

        return x_ref, x_dist, y
