
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
import os
import sys
import pickle
import time

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

class BackboneWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_type = config["model_type"]

        if self.model_type == "vanila_transformer":
            from backbones.vanila_transformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "efficient_attention":
            from backbones.efficient_attention import Backbone
            self.backbone = Backbone(config)

    def forward(self, X, mask):
        return self.backbone(X, mask)
