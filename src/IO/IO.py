"""File of class with input/output responsibilities"""

import os
import json
import numpy as np

class IO():
    """
    Class with input/output responsibilities
    """

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
        if not os.path.exists(directory):
            os.makedirs(directory)


    @staticmethod
    def json_dump(data_out, output_directory, filename):
        """
        Save data in json format

        Parameters
        ----------
        data_out: serializable object
            Data to be saved into json using json.dumps()
        """
        with open(os.path.join(output_directory,filename), 'w', encoding='utf8') as json_file:
            json_file.write(json.dumps(data_out))


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
                peak_seq_file.write('>' + sequence_names[i] + '\n')
                peak_seq_file.write(sequences[i] + '\n')
