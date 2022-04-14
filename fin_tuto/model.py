from numpy import dstack
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

class GNN(nn.Module):
  def __init__(self, feature_size, model_params):
    super(GNN, self).__init__()


    self.conv_layers = nn.ModuleList([])
    self.transf_layers = ModuleList([])
    self.pooling_layers = ModuleList([])
    self.bn_layers = ModuleList([])