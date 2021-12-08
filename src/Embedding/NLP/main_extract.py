
import numpy as np

from utils.utils import open_file, get_line, get_id
from utils.extract_embeddings import embeddings_and_vocab, texts_vector

pathResults = "data/results.npy"
pathTexts = "data/abstracts.txt"
pathModel = "weights/cbow_own4/model.pt"
pathVocab = "weights/cbow_own4/vocab.pt"

rawFile = open_file(pathTexts)
texts = [get_line(line) for line in rawFile]
ids = [get_id(line) for line in rawFile]

embeddings, vocab = embeddings_and_vocab(pathModel, pathVocab)

print(embeddings[0])
vectors = texts_vector(texts, embeddings, vocab)

results = np.zeros((len(texts), vectors.shape[1]+1))

results[:,0] = ids
results[:,1:] = vectors

np.save(pathResults, results)

