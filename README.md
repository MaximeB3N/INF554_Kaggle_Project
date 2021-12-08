# INF554_Kaggle_Project
Link for abstracts.txt file (1G)
https://www.dropbox.com/s/2s3lmm7utjbkg5e/abstracts.txt?dl=0

## Project Structure
```
.
├── README.md
├── config.yaml
├── requirements.txt
├── configs
    ├── config_gcn.yaml
    ├── config_abstracts.yaml

├── data
        ├── rawData
            ├── abstracts.txt
            ├── author_papers.txt
            ├── coauthorship.edgelist
            ├── train.csv
            ├── test.csv

        ├── embeddedData

├── trained_models
    ├── GCN
    ├── W2V
    ├── Regressors

├── src
    ├── Embedding
        ├── NLP
            ├── notebooks
                ├── Exploration.ipynb
                ├── Inference.ipynb
            ├── utils
                ├── constants.py
                ├── dataloader.py
                ├── datasets.py
                ├── extract_embeddings.py
                ├── helper.py
                ├── model.py
                ├── trainer.py
                ├── utils.py
            ├── main_extract.py
            ├── train.py
            ├── utils.py
            ├── config.yaml

        ├── Graph
            ├── deepWalk.py

    ├── Models
        ├── GCN
            ├── dataloader.py
            ├── gcn.py
            ├── utils.py

        ├── Regressors

    ├── utils
        ├── utils.py

    ├── stack_vectors.py
    ├── train_deepwalk.py
    ├── train_gcn.py
    ├── train_regressors.py
    ├── train_W2V.py
    
```
## Installation
In order to have the good environnement to run this code you need to :
- Create an virtual environnement (optional)
```
python3 -m venv venv
source venv/bin/activate
```

- Install all the needed dependencies
```
pip install -r requirements.txt
```

## Usage
### Text embeddings
```
python3 train_W2V.py --config configs/config_abstracts.py
```

We provide two possibilities to do a regression. The first one is extracting vectors from the text and the graph,
then use a regression model. The second one is using the abstracts to bring more information to the graph, then use
a graph neural network to do the regression.

### Possibility 1: Regressor on text embeddings and graph embeddings
#### Graph Embeddings
Find a graph representation using random walks and a Word2Vec model to 'learn' the graph. 
```
python3 train_deepwalk.py
```

#### Concatenate graph and text embeddings
Preprocess of the two previous step to have a matrix stacking the two representations. The matrix is in a convenient format for the next step.
```
python3 stack_vectors.py
```
#### Train regressors on stacked vectors
Train and test a bunch of models in order to find the best
```
python3 train_regressors.py
```

### Possibility 2: Text embedding and GCN with the graph
Train a GCN using both the relation graph and information found in the abstracts of papers
```
python3 train_gcn.py --config configs/config_gcn.py
```