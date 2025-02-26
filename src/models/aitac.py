"""Main AITAC model"""
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

# Convolutional neural network
class AITAC(nn.Module):
    """Main AITAC model
    """
    def __init__(self, num_classes: int, num_filters: int) -> None:
        """Main AITAC model
        """
        super().__init__()
        self.num_filters = num_filters
        # for layer one, separate convolution and relu step from maxpool and batch normalization
        # to extract convolutional filters
        self.layer1_conv = nn.Sequential(
            # padding is done in forward method along 1 dimension only,
            # Conv2D would do in both dimensions
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filters,
                      kernel_size=(4, 19), # 4 channel, 19 long sequences
                      stride=1,
                      padding=0),
            nn.ReLU())
        self.layer1_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(self.num_filters))
        self.layer2_conv = nn.Sequential(
            # padding is done in forward method along 1 dimension only,
            # Conv2D would do in both dimensions
            nn.Conv2d(in_channels=self.num_filters,
                      out_channels=200,
                      kernel_size=(1, 11),
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))
        self.layer3_conv = nn.Sequential(
            # padding is done in forward method along 1 dimension only,
            # Conv2D would do in both dimensions
            nn.Conv2d(in_channels=200,
                      out_channels=200,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))
        self.layer4_lin = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))
        self.layer5_lin = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))
        self.layer6_lin = nn.Sequential(
                nn.Linear(in_features=1000,
                          out_features=num_classes))#,
                #nn.Sigmoid())


    def forward(self, data_in: torch.Tensor) -> Tuple:
        """Forward pass
        Parameters
        ----------
        data_in: np.array
            Input data to train on dimensions?
        """
        # run all layers on input data
        # add dummy dimension to input (for num channels=1)
        data_in = torch.unsqueeze(data_in, 1)
        # Run convolutional layers
        # padding - last dimension goes first,
        # done here so that it is added along one dimension only
        data_in = F.pad(data_in, (9, 9), mode='constant', value=0)
        out = self.layer1_conv(data_in)
        activations = torch.squeeze(out)
        out = self.layer1_process(out)
        out = F.pad(out, (5, 5), mode='constant', value=0)
        out = self.layer2_conv(out)
        out = F.pad(out, (3, 3), mode='constant', value=0)
        out = self.layer3_conv(out)
        # Flatten output of convolutional layers
        out = out.view(out.size()[0], -1)
        # run fully connected layers
        out = self.layer4_lin(out)
        out = self.layer5_lin(out)
        predictions = self.layer6_lin(out)
        activations, act_index = torch.max(activations, dim=2)

        return predictions, activations, act_index
