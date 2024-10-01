import torch.nn as nn
from pdb import set_trace as st
import torch
import torch.nn.functional as F




class FCN(nn.Module):
    def __init__(self,input_size,weights_bias=[None]):
        super(FCN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size,64),
            nn.Tanh(),
            nn.Linear(64,1600),
        )


    def forward(self, x):

        x = self.layer1(x)

        x = torch.abs(x)

        return x




