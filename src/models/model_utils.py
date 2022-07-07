"""Class with static methods for model utils"""
import numpy as np
import torch
from torch import nn

class ModelUtils():
    """Collection of utility functions"""

    @staticmethod
    def centering_data(data_in: np.array) -> np.array:
        """
        Center incoming data
        """
        data_in_mean = torch.mean(data_in, dim=1, keepdim=True)
        centered_data = data_in - data_in_mean
        return centered_data


    @staticmethod
    def pearson_loss(data_in_1: np.array, data_in_2: np.array) -> int:
        """
        Define Pearson loss calculated as centered cosine similarity
        """

        centered_1 = ModelUtils.centering_data(data_in_1)
        centered_2 = ModelUtils.centering_data(data_in_2)
    
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.sum(1-cos(centered_1, centered_2))
        return loss 