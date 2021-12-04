import numpy as np

pathVectors = "data/vectors.npy"
vectors = np.load(pathVectors)

zeros = np.zeros(vectors.shape[1]-1)

#print(vectors.shape)
#print(vectors[0])
#print(np.all(vectors[:,1:]==zeros, axis=1))
#print((vectors[:,1:]==zeros).shape)
count = np.sum(np.all(vectors[:,1:]==zeros, axis=1))

print(f"Nb of [0,...,0] : {vectors.shape[0] - count}")
print(f"Prop of non null vectors : {1 - count/vectors.shape[0]}") 