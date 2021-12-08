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
    ├── Embedding
    
    ├── 
    
    ├──
    

├── train.py
├── utils
│   ├── constants.py
│   ├── dataloader.py
│   ├── helper.py
│   ├── model.py
│   └── trainer.py
└── weights
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation

- **train.py** - script for training
- **config.yaml** - file with training parameters
- **weights/** - folder where expriments artifacts are stored
- **notebooks/Inference.ipynb** - demo of how embeddings are used

## Usage

```

````


```
python3 train.py --config config.yaml
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram", "cbow")
- dataset ("WikiText2", "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")


