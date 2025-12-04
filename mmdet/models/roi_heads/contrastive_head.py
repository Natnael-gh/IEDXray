# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet.registry import MODELS

# @MODELS.register_module()
# class ContrastiveHead(nn.Module):
#     def __init__(self, in_channels=1024, hidden_dim=256, proj_dim=256):
#         super().__init__()  #i inherit from nn.Module
#         self.proj = nn.Sequential(  # anetwork with two linear layers and a ReLU activation in between to project features into a contrastive space
#             nn.Linear(in_channels, hidden_dim) ,
#             nn.ReLU(inplace=True), 
#             nn.Linear(hidden_dim, proj_dim)
#         )
#     def forward(self, x):
#         return F.normalize(self.proj(x), dim=-1) # L2 normalization along last dimension
    
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class ContrastiveHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_dim=256, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        # x should be [N, C]; we L2-normalize along the feature dimension
        return F.normalize(self.proj(x), dim=-1)
