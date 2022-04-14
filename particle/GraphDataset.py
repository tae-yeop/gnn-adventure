import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools

import numpy as np
import uproot
import glob
import multiprocessing

from pathlib import Path
import yaml

from tqdm.notebook import tqdm
import awkward as ak
from utils import get_file_handler


class GraphDataset(Dataset):
  def __init__(self, root, features, labels, spectators, transform=None, pre_transform=None,
               n_events=-1, n_events_merge=1000, file_names=None, remove_unlabeled=True):

    self.features = features
    self.labels = labels
    self.spectators = spectators
    self.n_events = n_events
    self.n_events_merge = n_events_merge
    self.file_names = file_names
    self.remove_unlabeled = remove_unlabeled
    super(GraphDataset, self).__init__(root, transform, pre_transform)

  