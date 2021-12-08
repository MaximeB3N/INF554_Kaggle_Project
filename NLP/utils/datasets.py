import torch

from utils.utils import get_line, open_file

PATH="data/abstracts.txt"

class datasets(torch.utils.data.Dataset):
    """"""
    def __init__(self, root=PATH, ds_type="train"):
        """"""
        self.ds_type = ds_type
        self.root = root
        self.data = [get_line(line) for line in open_file(root)]
        print(f"The dataset got {len(self.data)} elements")
        
    def __getitem__(self, index):
        """"""
        return self.data[index]
    
    def __len__(self):
        """"""
        return len(self.data)
