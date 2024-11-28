from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import skimage.color as color
import torch
from . import util
import os

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, HW=(256, 256), resample=3):
        """
        Dataset for colorization using the preprocess_img function.
        Args:
            root_dir (str): Directory with images.
            HW (tuple): Target height and width for resizing.
            resample (int): Resampling method for resizing (default: 3 for bicubic).
        """
        self.root_dir = root_dir
        self.HW = HW
        self.resample = resample
        self.image_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_rgb = np.array(Image.open(img_path).convert("RGB"))  # Load as RGB
        img = util.load_img(img_path)
        #print(f"IMG shape: {img.shape}")
        # Use preprocess_img to get L channels
        (tens_l_rs, tens_l_rs) = util.preprocess_img(img, HW=(256,256))
        tens_l_rs = tens_l_rs.squeeze(0)
        #print(f"RS shape: {tens_l_rs.shape}")
        # Convert ground truth ab channels
        img_lab = color.rgb2lab(img_rgb)
        img_ab = img_lab[:, :, 1:] / 128.0  # Normalize ab channels to [-1, 1]

        # Resize each channel of img_ab separately
        img_ab_resized = np.zeros((*self.HW, 2))  # Initialize resized array
        for i in range(2):  # Loop over a and b channels
            img_ab_resized[..., i] = np.asarray(
                Image.fromarray((img_ab[..., i] * 255).astype(np.uint8)).resize(self.HW, resample=self.resample)
            ) / 255.0  # Normalize back to [0, 1]

        tens_ab = torch.tensor(img_ab_resized, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        #print(f"RS shape2: {tens_l_rs.shape}")

        return {"L_resized": tens_l_rs, "AB_resized": tens_ab}


def get_data_loader(batch_size, root_dir, HW=(256, 256), shuffle=True, num_workers=4):
    """
    Returns a dataloader for the ColorizationDataset.
    """
    dataset = ColorizationDataset(root_dir, HW=HW)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
