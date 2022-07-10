import torch
from torch.utils import data
import h5py
import numpy as np
import SimpleITK as sitk


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
        img = img[0]
        
        if self.transform:
            img = self.transform(img)
        
        path = self.dir_list[idx]
        
        values_view = self.labels[idx].values()
        value_iterator = iter(values_view)
        label = next(value_iterator)
        
        return img, torch.tensor(label), path


class bodyPartDatasetMIL(data.Dataset):
    """Break_RD_dataset."""
    def __init__(self, dir_list, labels, patient, transform=None):
        super(bodyPartDatasetMIL, self).__init__()
        self.dir_list = dir_list
        self.labels = labels
        self.patient = patient
        self.transform = transform

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx): 
        img_list = []
        for i in range(len(self.dir_list[idx])):
            #print(self.dir_list[idx])
            image = sitk.ReadImage(self.dir_list[idx][i])
            img = sitk.GetArrayFromImage(image)
            img = img[0]

            if self.transform:
                img = self.transform(img)
                
            img_list.append(img)
            
        values_view = self.labels[idx].values()
        value_iterator = iter(values_view)
        label = next(value_iterator)

        return  img_list, torch.tensor(label), self.patient[idx]
