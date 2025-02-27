import numpy as np
import huggingface_hub
import os
from datasets import load_dataset
import pickle
import pandas as pd

state = os.environ['state']
np.random.seed(state)

huggingface_hub.login(token= os.environ['huggingface_token'])
dataset = load_dataset("DS23-KI-Projekt/alzheimerdataset_split")

df = dataset['train'].to_pandas()

if not os.path.exists('dataset.pkl'):
    os.makedirs('dataset.pkl')

pd.to_pickle(df, "dataset.pkl") 

