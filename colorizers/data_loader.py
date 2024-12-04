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
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample for training.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - "L_resized": Resized L channel tensor, shape [1, H, W].
                - "AB_resized": Resized AB channels tensor, shape [2, H, W].
        """
        # Load the image as a NumPy array
        img_path = self.image_files[idx]
        img_rgb = np.array(Image.open(img_path).convert("RGB"))  # Load as RGB
        img_rgb = util.resize_img(img_rgb, HW=self.HW)

        # Preprocess to get L channels
        tens_l_orig, tens_l_rs = util.preprocess_img(img_rgb, HW=self.HW)

        # Convert the resized RGB image to LAB and normalize AB channels
        img_lab = color.rgb2lab(img_rgb)  # Convert to LAB color space
        img_ab = img_lab[:, :, 1:] / 128.0  # Normalize AB channels to [-1, 1]

        # Convert AB channels to torch tensor (no resizing needed)
        tens_ab = torch.tensor(img_ab, dtype=torch.float32).permute(
            2, 0, 1
        )  # HWC to CHW

        # Remove batch dimension from L channel
        tens_l_rs = tens_l_rs.squeeze(0)
        return {"L_resized": tens_l_rs, "AB_resized": tens_ab}


def get_data_loader(
    batch_size, root_dir, HW=(256, 256), shuffle=True, num_workers=4
):
    """
    Returns a dataloader for the ColorizationDataset.
    """
    dataset = ColorizationDataset(root_dir, HW=HW)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
