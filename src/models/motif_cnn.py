"""File with class for extracting motifs from trained AITAC weights"""

from typing import Tuple
import torch
import torch.nn.functional as F
import torch.utils.data

from src.models.aitac import AITAC

# define model for extracting motifs from first convolutional layer
# and determining importance of each filter on prediction

# why not inherit and override from AITAC?
class MotifExtractionCNN(AITAC):
    """
    Extracting motifs from the trained (AITAC) model
    """

    def forward(self, data_in: torch.Tensor) -> Tuple:
        """
        Recreate forward pass as in AITAC, but also including explanation
        """
        # add dummy dimension to input (for num channels=1)
        data_in = torch.unsqueeze(data_in, 1)
        # Run convolutional layers
        # padding - last dimension goes first
        data_in = F.pad(data_in, (9, 9), mode='constant', value=0)
        out = self.layer1_conv(data_in)
        layer1_activations = torch.squeeze(out)
        # do maxpooling and batch normalization for layer 1
        layer1_out = self.layer1_process(out)
        layer1_out = F.pad(layer1_out, (5, 5), mode='constant', value=0)
        # calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1)
        # run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        # 81 = number of classes
        predictions = torch.zeros(batch_size, self.num_filters,  81)
        for i in range(self.num_filters):
            # modify filter i of first layer output
            filter_input = layer1_out.clone()
            # num_channels = 1
            # why 94?
            filter_input[:,i,:,:] = filter_input.new_full(
                (batch_size, 1, 94),
                fill_value = filter_means_batch[i])
            out = self.layer2_conv(filter_input)
            out = F.pad(out, (3, 3), mode='constant', value=0)
            out = self.layer3_conv(out)
            # Flatten output of convolutional layers
            out = out.view(out.size()[0], -1)
            # run fully connected layers
            out = self.layer4_lin(out)
            out = self.layer5_lin(out)
            out = self.layer6_lin(out)
            predictions[:,i,:] = out
            # find index of highest activation layer
            _, act_index = torch.max(layer1_activations, dim=2)

        return predictions, layer1_activations, act_index
           