"""File to provide entry points to user interaction"""

import numpy as np
import torch

from src.data_processing.preprocess_data import Preprocessor
from src.utils.IO import IO
from src.utils import plot_utils
from src.models import train_test_aitac

class UserInterface:
    """
    Class to provide entry points to user interaction
    """

    def run_preprocessing(self):
        """
        Method to run preprocessing steps
        """
        my_io = IO(
            "config.yml"
            )
        data_file = my_io.get_config_entry("data_file")
        intensity_file = my_io.get_config_entry("intensity_file")
        reference_genome_dir = my_io.get_config_entry("reference_genome_dir")
        output_directory = my_io.get_config_entry("output_directory")
        num_chr = my_io.get_config_entry("num_chr")
        region_size = my_io.get_config_entry("region_size")
        my_io.log_to_file("initialized Preprocessor")
        my_preprocessor = Preprocessor(
            data_file,
            intensity_file,
            reference_genome_dir,
            output_directory,
            num_chr,
            region_size)
        my_io.log_to_file("read data")
        positions, chr_dict = my_preprocessor.read_in_data()
        my_io.log_to_file("reformat data")
        my_preprocessor.reformat_data(positions, chr_dict)
        my_io.log_to_file("done")

    def train_main_model(
            self,
            model_name: str,
            outdir: str,
            config_path: str,
            x_file:str,
            y_file: str,
            peak_names_file:str) -> None:
        """
        Train AITAC model
        :param model_name: title of analysis
        :param x_file: path to the data file - without validation data
        :param y_file: path to the label file - without validation data
        :param peak_names_file: path to the peak names file
        """
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("set hyperparams")
        # Set hyper parameters
        num_epochs = 10
        num_classes = 81
        batch_size = 100
        learning_rate = 0.001
        num_filters = 300
        print("load data")
        # Load all data
        x = np.load(x_file)
        x = x.astype(np.float32)
        y = np.load(y_file)
        y = y.astype(np.float32)
        peak_names = np.load(peak_names_file)
        print("create model instance")
        tt_instance = train_test_aitac.TrainTestModel()
        tt_instance.create_out_directory(model_name)
        print("create data loader")
        train_loader, eval_loader, eval_labels, eval_names = tt_instance.create_data_loader(
            x,
            y,
            peak_names,
            batch_size)
        print("fit model instance")
        model, _ = tt_instance.fit(
            model_name,
            num_classes,
            num_filters,
            train_loader,
            eval_loader,
            learning_rate,
            num_epochs,
            config_path,
            device)
        print("eval fit")
        predictions, max_activations, max_act_index = tt_instance.eval(eval_loader, model,device)
        print("Creating plots...")
        # plot the correlations histogram
        correlations = plot_utils.plot_cors(eval_labels, predictions, outdir)
        # returns correlation measurement for every prediction-label pair
        plot_utils.plot_corr_variance(eval_labels, correlations, outdir)
        quantile_indx = plot_utils.plot_piechart(correlations, eval_labels, outdir)
        plot_utils.plot_random_predictions(
            eval_labels,
            predictions,
            correlations,
            quantile_indx,
            eval_names,
            outdir)
        print("save output")
        #save predictions
        np.save(outdir + "predictions.npy", predictions)
        #save correlations
        np.save(outdir + "correlations.npy", correlations)
        #save max first layer activations
        np.save(outdir + "max_activations.npy", max_activations)
        np.save(outdir + "max_activation_index.npy", max_act_index)
        #save test data set
        np.save(outdir + "test_OCR_names.npy", eval_names)
