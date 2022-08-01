"""File of class with input/output responsibilities"""

import os
import json
import _pickle as pickle
import yaml
import numpy as np

class IO:
    """
    Class with input/output responsibilities
    """

    def __init__(self, config_file) -> None:
        """
        Constructor for internalizing config file

        Parameters
        ----------
        config_file: str
            Path to config
        """
        self.config = self.load_config_file(config_file)


    def load_config_file(self, config_file):
        """
        Readin in config file

        Parameters
        ----------
        config_file: str
            Path to config

        Return
        ------
        config: dict
            Dictionary of config entries
        """
        with open(config_file, "r", encoding='utf8') as file:
            config = yaml.safe_load(file)
        return config


    def get_config_entry(self, config_entry):
        """
        Getter for a config entry

        Return
        ------
        _ :str
            Config entry
        """
        return self.config[config_entry]


    def log_to_file(self, message):
        """
        Log message
        """
        with open(self.config["log"], "a", encoding='utf8') as log_file:
            log_file.write(message + "\n")


    def write_to_file(self, filename: str, message: str) -> None:
        """
        Write any message to any file
        """
        out_path = os.path.join(self.config["output_directory"], filename)
        with open(out_path, "a", encoding='utf8') as out_file:
            out_file.write(message + "\n")


    @staticmethod
    def create_directory_if_not_exists(directory_path):
        """
        Check if directory exists, if not, create directory

        Parameters
        ----------
        directory_path: str
            Path to directory to create
        """
        directory = os.path.dirname(directory_path)
        os.makedirs(directory, exist_ok=True)


    @staticmethod
    def is_file_exists(directory, filename):
        """
        Check if file exists in directory

        Parameters
        ----------
        directory_path: str
            Path to directory with suspected file
        filename: str
            Suspected file

        Return
        ------
        _ : bool
            True if exists, False if not
        """
        return bool(os.path.exists(os.path.join(directory, filename)))


    @staticmethod
    def join_path(directory, filename):
        """
        Wrapper around os.path.join
        """
        return os.path.join(directory, filename)


    @staticmethod
    def save_data_in_json(data_out, output_directory, filename):
        """
        Save data in json format

        Parameters
        ----------
        data_out: serializable object
            Data to be saved into json using json.dumps()
        output_directory: str
            Path to the output folder
        filename: str
            Output filename
        """
        with open(os.path.join(output_directory,filename+'.json'), 'w', encoding='utf8') as json_file:
            json_file.write(json.dumps(data_out))


    @staticmethod
    def save_data_in_pickle(data_out, output_directory, filename):
        """
        Save data in pickle format

        Parameters
        ----------
        data_out: serializable object
            Data to be saved into pickle using pickle.dump()
        output_directory: str
            Path to the output folder
        filename: str
            Output filename
        """
        with open(os.path.join(output_directory,filename+'.pickle'), "wb") as pickle_file:
            pickle.dump(data_out, pickle_file)


    @staticmethod
    def numpy_save(numpy_data_out, output_directory, filename):
        """
        Save numpy data in an npy format

        Parameters
        ----------
        numpy_data_out: numpy array
            Data to be saved into a .npy file using np.save()
        output_directory: str
            Path to the output folder
        filename: str
            Output filename
        """
        np.save(os.path.join(output_directory, filename + '.npy'), numpy_data_out)


    @staticmethod
    def write_seq_to_fasta(sequences, sequence_names, output_directory, filename):
        """
        Save sequences to fasta file

        Parameters
        ----------
        filename: str
            Output filename
        sequences: numpy.ndarray
            Sequences of input regions (num_positions x seq_length ?? )
        sequence_names: numpy.ndarray
            Names of valid input regions (num_valid_positions x name_length ?? )
        """
        filename = os.path.join(output_directory, filename + '.fasta')
        with open(filename, 'w', encoding='utf8') as peak_seq_file:
            for i in range(sequences.shape[0]):
                peak_seq_file.write('>peak_' + str(sequence_names[i]) + '\n')
                peak_seq_file.write(sequences[i] + '\n')


    @staticmethod
    def read_pickle(output_directory, filename):
        """
        Read data from pickle format

        Parameters
        ----------
        output_directory: str
            Path to the output folder
        filename: str
            Output filename

        Return
        ------
        data_in: serializable object
            Data read from pickle using pickle.load()
        """
        with open(os.path.join(output_directory,filename+'.pickle'), "rb") as pickle_file:
            data_in = pickle.load(pickle_file)
        return data_in
