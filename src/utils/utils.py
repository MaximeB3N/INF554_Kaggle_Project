import numpy as np


def get_authors_to_papers(pathAuthorsPapers, vectors, id_abstracts_num):

    ## Getting relation between authors and their papers
    with open(pathAuthorsPapers) as f:
        authors_papers=f.readlines()

    # Creating a matrix that will contain {author ID, vector} with the vector being the normalized sum of all its papers vectors
    n_dim_abstracts_vectors=vectors.shape[1]-1
    authors_vectors=np.zeros((len(authors_papers), n_dim_abstracts_vectors+1), dtype=np.float64)

    # Processing author-paper relation
    papers=authors_papers[0].split("\n")[0].split(":")[1].split("-")
    s=0
    for i,author in enumerate(authors_papers):
        papers=author.split("\n")[0].split(":")[1].split("-")
        vector=np.zeros(n_dim_abstracts_vectors)
        no_fail=False
        for p in papers:
            try:
                vector+=vectors[id_abstracts_num[int(p)], 1:]
                no_fail=True
            except KeyError:
                pass
        if (not no_fail):
            s+=1
        authors_vectors[i][0]=int(author.split(":")[0])
        if (np.linalg.norm(vector)>0):
            vector=vector/np.linalg.norm(vector)
        authors_vectors[i][1:]=vector.copy()
    print(s) # Checking how many authors have no abstracts in our database
