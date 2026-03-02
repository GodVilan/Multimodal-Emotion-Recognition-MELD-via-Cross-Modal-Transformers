import os
import torch
from torch.utils.data import Dataset


class MELDDataset(Dataset):

    def __init__(self, processed_folder):
        self.processed_folder = processed_folder
        self.files = sorted(os.listdir(processed_folder))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_folder, self.files[idx])
        return torch.load(file_path)
