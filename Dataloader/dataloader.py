import os
import numpy as np
import torch
import pandas as pd

from Dataloader.training import load_vectorized_text_numpy_data, get_set, get_indices

def create_dataloader(pathData, pathLabels, batch_size, num_workers=os.cpu_count(),shuffle=True):
    
    dataset = Dataset(pathData, pathLabels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        shuffle=shuffle, num_workers=num_workers)
    return dataloader



class Dataset(torch.utils.data.Dataset):  
    def __init__(self, pathData, pathLabels):
        ids, vectors = load_vectorized_text_numpy_data(pathData)
        
        df = pd.read_csv(pathLabels)

        vectors_set = get_set(ids)

        indices = get_indices(df['author'].to_numpy(),vectors_set)

        self.inputs = vectors[indices,:]
        self.hindex = df['hindex'].to_numpy()

    
    # def __init__(self, pathNLP, pathGraph, pathLabels):    
    #     Abstract_ids, Abstract_vec = load_vectorized_text_numpy_data(pathNLP)
    #     Graph_ids, Graph_vec = load_vectorized_text_numpy_data(pathGraph)

    #     df = pd.read_csv(pathLabels)

    #     self.inputs = np.zeros((len(df), Abstract_vec.shape[-1]+Graph_vec.shape[-1]))
    #     Abstract_set = get_set(Abstract_ids)
    #     Graph_set = get_set(Graph_ids)

    #     Abstract_indices = get_indices(df['author'].to_numpy(),Abstract_set)
    #     Graph_indices = get_indices(df['author'].to_numpy(),Graph_set)

    #     self.inputs[:,] = np.concatenate((Graph_vec[Graph_indices],
    #                                          Abstract_vec[Abstract_indices]),
    #                                          axis=1)
    #     self.hindex = df['hindex'].to_numpy()

    def __getitem__(self, index):
        return self.inputs[index], self.hindex[index]

    def __len__(self):
        return len(self.inputs)