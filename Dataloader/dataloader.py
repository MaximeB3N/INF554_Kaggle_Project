import torch
import os

from Dataloader.training import load_vectorized_text_numpy_data, find_input_datas


def create_dataloader(pathData, pathLabels, batch_size, num_workers=os.cpu_count(),shuffle=True):
    
    dataset = Dataset(pathData, pathLabels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        shuffle=shuffle, num_workers=num_workers)
    return dataloader



class Dataset(torch.utils.data.Dataset):  
    def __init__(self, pathData, pathLabels):
        X, ids = load_vectorized_text_numpy_data(pathData)
        self.inputs, self.labels = find_input_datas(pathLabels, X, ids)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)