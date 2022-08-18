from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import matplotlib
import os
import pathlib

from src.models.model_utils import ModelUtils
matplotlib.use('Agg')

from src.models import aitac


class TrainTestModel:
    """
    Class for methods to run main model training and testing
    """

    def create_out_directory(self, model_name : str) -> None:
        """Create output figure directory
        :param model_name: unique ID for the analysis
        """
        output_file_path = "outputs/" + model_name + "/training/"
        print(f"Creating directory {output_file_path}")
        outdir = os.path.dirname(output_file_path)
        os.makedirs(outdir, exist_ok=True)
        # save the model checkpoint
        models_file_path = "my_models/"
        models_directory = os.path.dirname(models_file_path)
        print(f"Creating directory {models_file_path}")
        os.makedirs(models_directory, exist_ok=True)

        return

    def create_data_loader(
            self,
            X:np.ndarray,
            y:np.ndarray,
            peak_names:np.ndarray,
            batch_size: int,
            test_size:float =0.1) -> Tuple:
        """
        Load data
        """
        # split the data into training and test sets
        print("do train test split")
        train_data, eval_data, train_labels, eval_labels, _, eval_names = train_test_split(
            X,
            y,
            peak_names,
            test_size=test_size,
            random_state=40)
        # Data loader
        print("create dataset")
        train_dataset = TensorDataset(
            torch.from_numpy(train_data),
            torch.from_numpy(train_labels))
        print("create data loader")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False)
        print("create dataset")
        eval_dataset = TensorDataset(
            torch.from_numpy(eval_data),
            torch.from_numpy(eval_labels))
        print("create data loader")
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False)

        return train_loader, eval_loader, eval_labels, eval_names


    def fit(
            self,
            model_name: str,
            num_classes:int,
            num_filters: int,
            train_loader: DataLoader,
            eval_loader: DataLoader,
            learning_rate: float,
            num_epochs: int,
            config_path: str,
            device:torch.device) -> Tuple:
        """
        Function to fit the model
        This should be restructured and follow torch pattern
        """
        # create model
        model = aitac.AITAC(num_classes, num_filters).to(device)
        # Loss and optimizer
        criterion = ModelUtils.pearson_loss
        optimizer = Adam(model.parameters(), lr=learning_rate)
        # train model
        model, best_loss = ModelUtils.train_model(
            train_loader,
            eval_loader,
            model,
            device,
            criterion,
            optimizer,
            num_epochs,
            config_path)
        # save weigths ToDo: move to a different function
        torch.save(model.state_dict(), 'my_models/' + model_name + '.ckpt')
        #save the whole model
        torch.save(model, 'my_models/' + model_name + '.pth')

        return model, best_loss


    def eval(
            self,
            eval_loader:DataLoader,
            model:nn.Module,
            device:torch.device) -> None:
        """
        Predict on eval set
        """
        # Predict on test set
        predictions, max_activations, max_act_index = ModelUtils.test_model(
            eval_loader,
            model,
            device)

        return predictions, max_activations, max_act_index
