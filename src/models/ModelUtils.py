"""File to define utility functions used for models"""
import torch
from torch import nn

class ModelMethods():
    """
    Class to define utility functions
    """

    @staticmethod
    def center_vector(vector_in):
        """
        Center values around 0
        """
        mean_vector = torch.mean(vector_in, dim=1, keepdim=True)
        centered_vector = vector_in - mean_vector
        return centered_vector


    @staticmethod
    def pearson_loss(predicted, ground_truth):
        """
        Calculate Pearson correlation as cosine similarity between centered vectors
        """
        # this thread says highly noisy regression is a good use case for correlation as a loss
        # the same thread says correlation is good if you are only interested in the shape
        # MSE is good if you are interested in the actual values
        # https://stats.stackexchange.com/questions/228373/use-pearsons-correlation-coefficient-as-optimization-objective-in-machine-learn
        centered_predicted = ModelMethods.center_vector(predicted)
        centered_truth = ModelMethods.center_vector(ground_truth)

        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.sum(1-cosine_similarity(centered_predicted,centered_truth))
        return loss
