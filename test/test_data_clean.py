import os
from huggingface_hub import login
from datasets import load_dataset
import numpy as np
import pytest

def test_data_null():
    login(token= os.environ['huggingface_token'])
    dataset = load_dataset("DS23-KI-Projekt/alzheimerdataset_split")
    df = dataset["train"].to_pandas()
    
    assert df.isna().sum().sum() == 0

def test_data_dublicate():
    login(token= os.environ['huggingface_token'])
    dataset = load_dataset("DS23-KI-Projekt/alzheimerdataset_split")
    df = dataset["train"].to_pandas()
    assert df.shape[0]-df.drop_duplicates().shape[0] == 0


