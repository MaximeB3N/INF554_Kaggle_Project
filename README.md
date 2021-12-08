# INF554_Kaggle_Project
Link for abstracts.txt file (1G)
https://www.dropbox.com/s/2s3lmm7utjbkg5e/abstracts.txt?dl=0

## Project Structure


```
.
├── README.md
├── config.yaml
├── notebooks
│   └── Inference.ipynb
├── requirements.txt
├── code
    ├── data
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
        ├── deepWalk.py
        ├── fullEmbedding.py
    ├── Models
        ├── GCN
        ├── Regressors
    ├── trained_models
    ├── utils
    ├── train_deepwalk.py
    ├── train_gcn.py
    ├── train_regressors.py
    
```

## Usage

```
python3 stack_vectors.py
```

```
python3 train_deepwalk.py
```

```
python3 train_regressors.py
```

```
python3 train_gcn.py --config Models/GCN/train_gcn/config_gcn.py
```
