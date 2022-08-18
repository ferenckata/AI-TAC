"""Class with static methods for model utils"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from src.utils.IO import IO

class ModelUtils:
    """Collection of utility functions"""

    @staticmethod
    def centering_data(data_in: torch.Tensor) -> torch.Tensor:
        """
        Center incoming data

        Parameters
        ----------
        data_in: torch.Tensor

        Return
        ------
        centered_data: torch.Tensor
            Data with mean=0, variance = var(data_in)
        """
        data_in_mean = torch.mean(data_in, dim=1, keepdim=True)
        centered_data = data_in - data_in_mean
        return centered_data


    @staticmethod
    def pearson_loss(data_in_1: torch.Tensor, data_in_2: torch.Tensor) -> torch.Tensor:
        """
        Define Pearson loss calculated as centered cosine similarity

        Parameters
        ----------
        data_in_1: torch.Tensor
        data_in_2: torch.Tensor

        Return
        ------
        P_correlation: torch.Tensor
            Pearson correlation between data_in_1 and data_in_2
        """
        centered_1 = ModelUtils.centering_data(data_in_1)
        centered_2 = ModelUtils.centering_data(data_in_2)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        p_correlation = torch.sum(1-cos(centered_1, centered_2))
        return p_correlation


    @staticmethod
    def train_model(
                train_loader: DataLoader,
                test_loader: DataLoader,
                model: nn.Module,
                device: torch.device,
                criterion: nn.Module,
                optimizer: torch.optim,
                num_epochs: int,
                config_path: str) -> tuple:
        """Function to train model

        Parameters
        ----------
        train_loader: DataLoader
        test_loader: DataLoader
        model: nn.Module
        device: torch.device
        criterion: nn.Module
        optimizer: torch.optim
        num_epochs: int
        config_path: str

        Return
        ------
        """
        io_instance = IO(config_path)
        total_step = len(train_loader)
        model.train()

        #open files to log error
        train_error = "training_error.txt"
        test_error = "test_error.txt"

        best_model_weights = copy.deepcopy(model.state_dict())
        best_loss_valid = float('inf')

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (seqs, labels) in enumerate(train_loader):
                seqs = seqs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs, _, _ = model(seqs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    log_line = f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():{10}.{4}}'
                    io_instance.log_to_file(log_line)

            # save training loss to file
            epoch_loss = running_loss / len(train_loader.dataset)
            io_instance.write_to_file(train_error, f"{epoch}, {epoch_loss}")

            # calculate test loss for epoch
            test_loss = 0.0
            with torch.no_grad():
                model.eval()
                for i, (seqs, labels) in enumerate(test_loader):
                    input_x = seqs.to(device)
                    label_y = labels.to(device)
                    outputs, _, _ = model(input_x)
                    loss = criterion(outputs, label_y)
                    test_loss += loss.item()

            test_loss = test_loss / len(test_loader.dataset)

            # save outputs for epoch
            io_instance.write_to_file(test_error, f"{epoch}, {test_loss}")

            if test_loss < best_loss_valid:
                best_loss_valid = test_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                log_line = f'Saving the best model weights at Epoch [{epoch+1}], Best Valid Loss: {best_loss_valid:{10}.{4}}'
                io_instance.log_to_file(log_line)

        model.load_state_dict(best_model_weights)
        return model, best_loss_valid


    @staticmethod
    def test_model(test_loader, model, device):
        """
        Test trained model on test data
        """
        num_filters = model.layer1_conv[0].out_channels
        predictions = torch.zeros(0, 81)
        max_activations = torch.zeros(0, num_filters)
        act_index = torch.zeros(0, num_filters)

        with torch.no_grad():
            model.eval()
            for seqs, _ in test_loader:
                seqs = seqs.to(device)
                pred, act, idx = model(seqs)
                predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
                max_activations = torch.cat((max_activations, act.type(torch.FloatTensor)), 0)
                act_index = torch.cat((act_index, idx.type(torch.FloatTensor)), 0)

        predictions = predictions.numpy()
        max_activations = max_activations.numpy()
        act_index = act_index.numpy()
        return predictions, max_activations, act_index


    @staticmethod
    def get_motifs(data_loader, model, device):
        """
        Get motifs from trained model
        """
        num_filters=model.layer1_conv[0].out_channels
        activations = torch.zeros(0, num_filters, 251)
        predictions = torch.zeros(0, num_filters, 81)
        with torch.no_grad():
            model.eval()
            for seqs, _ in data_loader:
                seqs = seqs.to(device)
                pred, act, _ = model(seqs, num_filters)
                activations = torch.cat((activations, act.type(torch.FloatTensor)), 0)
                predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
        predictions = predictions.numpy()
        activations = activations.numpy()
        return activations, predictions
