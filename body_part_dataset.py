import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage.exposure import rescale_intensity


class bodyPartDataset(data.Dataset):
    """Break_RD_dataset."""
    def __init__(self, dir_list, labels, transform=None):
        super(bodyPartDataset, self).__init__()
        self.dir_list = dir_list
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.dir_list[idx])
        img = sitk.GetArrayFromImage(image)
        img = rescale_intensity(img, in_range='image', out_range=(0, 255))
        img = Image.fromarray((img[0]).astype(np.uint8), mode='L')
        # img = np.repeat(img.T, 3, -1).astype(np.uint8)
        # img = rescale_intensity(img, in_range='image', out_range=(0, 255))

        if self.transform:
            img = self.transform(img)

        img = np.repeat(img[...], 3, 0)
        path = self.dir_list[idx]
        values = self.labels[idx]
        # value_iterator = iter(values_view)
        # label = next(value_iterator)
        
        return img, torch.tensor(values), path

