import torch
from torch_geometric.data import DataLoader


import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://localhost:5000")



def run_one_training(params):
  params = params[0]
  with mlflow.start_run() as run:
    mlflow.