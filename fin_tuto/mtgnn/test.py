import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


x = torch.randn((1, 3, 5, 6))
A = torch.randn(4, 5)


t = torch.einsum('ncwl, vw -> ncvl', (x,A))
