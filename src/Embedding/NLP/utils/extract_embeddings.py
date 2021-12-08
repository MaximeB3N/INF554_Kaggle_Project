import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.dataloader import get_english_tokenizer


def embeddings_and_vocab(pathModel, pathVocab, device='cpu'):
    """
    Function that returns the embeddings of the data
    """
    
    model = torch.load(pathModel, map_location=torch.device(device))
    vocab = torch.load(pathVocab, map_location=torch.device(device))

    embeddings = list(model.parameters())[0]
    embeddings = embeddings.detach().numpy()

    return embeddings, vocab


def texts_vector(texts, embeddings, vocab):
    """
    Function that returns the embeddings of the texts
    """
    vectors = np.zeros((len(texts), 150))
    tokenizer = get_english_tokenizer()
    text_pipeline = lambda x: vocab(tokenizer(x))

    for i,text in enumerate(tqdm(texts)):

        text_tokens_ids = text_pipeline(text)
        vector = embeddings[text_tokens_ids].sum(axis=0)
        norms = (vector ** 2).sum() ** (1 / 2)
        vectors[i] = vector / norms

    return vectors