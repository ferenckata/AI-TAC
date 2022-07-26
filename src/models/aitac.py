"""Code for the main module of the system, the AI-TAC network"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

# Convolutional neural network
class ConvNet(nn.Module):
    """
    Defining the convolutional network structure
    """
    def __init__(self, num_classes, num_filters):
        super().__init__()

        # for layer one, separate convolution and relu step from maxpool and batch normalization
        # to extract convolutional filters
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(4, 19), # 4 channel, 19 long sequences
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU())

        self.layer1_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(num_filters))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters,
                      out_channels=200,
                      kernel_size=(1, 11),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=200,
                      out_channels=200,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.linear_last = nn.Sequential(
                nn.Linear(in_features=1000,
                          out_features=num_classes))#,
                #nn.Sigmoid())


    def forward(self, data_in):
        """
        Forward pass with padding for some reason
        """
        # add channel dimension to input so that it has [batch, channel, height, width]
        data_in = torch.unsqueeze(data_in, 1)

        # Run convolutional layers
        # padding the last dimension (width) from both sides
        # why here?
        data_in = F.pad(data_in, (9, 9), mode='constant', value=0)
        out = self.layer1_conv(data_in)
        activations = torch.squeeze(out)
        out = self.layer1_process(out)

        out = F.pad(out, (5, 5), mode='constant', value=0)
        out = self.conv2(out)

        out = F.pad(out, (3, 3), mode='constant', value=0)
        out = self.conv3(out)

        # Flatten output of convolutional layers
        out = out.view(out.size()[0], -1)

        # run fully connected layers
        out = self.linear1(out)
        out = self.linear2(out)
        predictions = self.linear_last(out)

        activations, act_index = torch.max(activations, dim=2)

        return predictions, activations, act_index

# define model for extracting motifs from first convolutional layer
# and determining importance of each filter on prediction
class MotifExtractionCNN(nn.Module):
    """
    Class to extract motifs
    """
    def __init__(self, original_model):
        """
        This seems to be the same as above, inheritance of some sort
        """
        super().__init__()
        self.layer1_conv = nn.Sequential(*list(original_model.children())[0])
        self.layer1_process = nn.Sequential(*list(original_model.children())[1])
        self.layer2 = nn.Sequential(*list(original_model.children())[2])
        self.layer3 = nn.Sequential(*list(original_model.children())[3])

        self.layer4 = nn.Sequential(*list(original_model.children())[4])
        self.layer5 = nn.Sequential(*list(original_model.children())[5])
        self.layer6 = nn.Sequential(*list(original_model.children())[6])


    def forward(self, data_in, num_filters):
        """
        same as above it seems, plus something about the filters
        """
        # duplicated code
        # add dummy dimension to input (for num channels=1)
        data_in = torch.unsqueeze(data_in, 1)

        # Run convolutional layers
        # padding - last dimension goes first
        data_in = F.pad(data_in, (9, 9), mode='constant', value=0)
        out= self.layer1_conv(data_in)
        layer1_activations = torch.squeeze(out)

        # do maxpooling and batch normalization for layer 1
        layer1_out = self.layer1_process(out)
        layer1_out = F.pad(layer1_out, (5, 5), mode='constant', value=0)

        # calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1)

        # run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        predictions = torch.zeros(batch_size, num_filters,  81)

        for i in range(num_filters):
            # modify filter i of first layer output
            filter_input = layer1_out.clone()

            filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=filter_means_batch[i])

            out = self.layer2(filter_input)
            out = F.pad(out, (3, 3), mode='constant', value=0)
            out = self.layer3(out)

            # Flatten output of convolutional layers
            out = out.view(out.size()[0], -1)

            # run fully connected layers
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)

            predictions[:,i,:] = out

            _, act_index = torch.max(layer1_activations, dim=2)

        return predictions, layer1_activations, act_index
