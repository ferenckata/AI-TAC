""" This module is for preprocessing input files for subsequent deep learning applications """

import numpy as np
from data_processing.preprocess_utils import PreprocessingMethods as PM
from IO.IO import IO as my_io

class Preprocessor():
    """
    A class used to read in an reformat data for the deep learning application

    ...

    Attributes
    ----------

    Methods
    -------


    """

    def __init__(self, data_file, intensity_file, reference_genome_dir, output_directory, num_chr, region_size):
        """
        Constructor

        Parameters
        ----------
        data_file: str
            Path to the location of the input bed file with peak coordinates
            The bed file should have four columns in this order: chr, start, end, name
        intensity_file: str
            Path to the tab (or whitespace) deliminated text file with normalized peak heights
        reference_genome_dir: str
            Path to the directory with reference genome fasta files
        output_directory: str
            Path to output folder where all intermediate files will be saved
        num_chr: int
            Number of autosomal chromosomes (X and Y will be searched for and used if available),
            this should match the name of the reference genome
            fasta file names in the following manner: chr{num_chr}.fa
        region_size: int
            Size of bed regions, the bed file should have entries of the same size
            region_size = (end - start)
        """
        self.data_file = data_file
        self.intensity_file = intensity_file
        self.reference_genome_dir = reference_genome_dir
        self.output_directory = output_directory
        self.num_chr = num_chr
        self.region_size = region_size
        my_io.create_directory_if_not_exists(self.output_directory)


    def read_in_data(self):
        """
        Function to extract data from file and save to file for later usage

        Return
        ------
        positions: dict
            Dictionary with keys: location IDs, and values: tuple of (chr, start, end)
        chr_dict: dict
            Dictionary of genomic sequences in Bio.SeqRecord format
            (one entry per fasta entry)
        """
        # read bed file with peak positions
        positions, skipped_peaks = PM.read_bed(self.data_file, self.region_size)
        # read reference genome fasta file into dictionary
        if not my_io.is_file_exists(self.output_directory,'chr_dict.pickle'):
            chr_dict = PM.read_fasta(self.reference_genome_dir, self.num_chr)
            my_io.save_data_in_pickle(chr_dict, self.output_directory, 'chr_dict')
        else:
            chr_dict = my_io.read_pickle(self.output_directory, 'chr_dict')

        return positions, chr_dict, skipped_peaks


    def reformat_data(self, positions, chr_dict):
        """
        Acquire encoded sequences and peak intensities of valid positions

        Parameters
        ----------
        positions: dict
            Dictionary with keys: location IDs, and values: tuple of (chr, start, end)
        chr_dict: dict
            Dictionary of genomic sequences in Bio.SeqRecord format
            (one entry per fasta entry)
            
        Return
        ------
        valid_peak_ids: numpy.ndarray
            Names of valid input regions (num_valid_positions x name_length ?? )
        one_hot_seqs: numpy.ndarray
            One-hot encoded sequences of input regions (num_positions x 4 x seq_length)
        peak_seqs: numpy.ndarray
            Sequences of input regions (num_positions x seq_length ?? )
        cell_type_array: numpy.ndarray
            Matrix of peak intensity values across cell type (num_positions x num_celltypes)
        invalid_ids: list
            Names of invalid input regions (num_invalid_positions x name_length ?? )
        """
        one_hot_seqs, peak_seqs, sequence_peak_names, invalid_ids = PM.get_sequences(positions,
                                                                                    chr_dict,
                                                                                    self.num_chr)
        
        # read in all intensity values and peak names
        cell_type_array, intensity_peak_names = PM.read_intensities(self.intensity_file)

        # take one_hot encoding of valid sequences of only those peaks that
        # have associated intensity values in cell_type_array
        valid_peak_ids = np.intersect1d(sequence_peak_names, intensity_peak_names)
        seq_data_values = PM.filter_matrix(sequence_peak_names, valid_peak_ids,
                                           one_hot_seqs, peak_seqs)
        sequence_peak_names = seq_data_values[0]
        one_hot_seqs = seq_data_values[1]
        peak_seqs = seq_data_values[2]

        peak_data_values = PM.filter_matrix(intensity_peak_names, valid_peak_ids,
                                            cell_type_array)
        intensity_peak_names = peak_data_values[0]
        cell_type_array = peak_data_values[1]

        # throw error here, add test for it
        if np.sum(sequence_peak_names != intensity_peak_names) > 0:
            raise AssertionError("Order of peaks not matching for sequences/intensities!")
        if np.sum(sequence_peak_names != valid_peak_ids) > 0:
            raise AssertionError("Order of peaks not matching for sequences/valid positions!")

        return valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids


    def save_data_to_file(self, valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids, skipped_peaks):
        """
        Function to save all the data created in this process

        Parameters
        ----------
        valid_peak_ids: numpy.ndarray
            Names of valid input regions (num_valid_positions x name_length ?? )
        one_hot_seqs: numpy.ndarray
            One-hot encoded sequences of input regions (num_positions x 4 x seq_length)
        peak_seqs: numpy.ndarray
            Sequences of input regions (num_positions x seq_length ?? )
        cell_type_array: numpy.ndarray
            Matrix of peak intensity values across cell type (num_positions x num_celltypes)
        invalid_ids: list
            Names of invalid input regions (num_invalid_positions x name_length ?? )
        """
        # write position and sequence data to file
        my_io.numpy_save(one_hot_seqs, self.output_directory, 'one_hot_seqs')
        my_io.numpy_save(valid_peak_ids, self.output_directory, 'peak_names')
        my_io.numpy_save(peak_seqs, self.output_directory, 'peak_seqs')
        my_io.numpy_save(cell_type_array, self.output_directory, 'cell_type_array')

        # save position names of invalid sequences
        my_io.save_data_in_json(invalid_ids, self.output_directory, 'invalid_ids')
        my_io.save_data_in_json(skipped_peaks, self.output_directory, 'peaks_of_invalid_length')

        # write peak sequences to fasta file
        out_seq_filename = 'peak_sequences'
        my_io.write_seq_to_fasta(peak_seqs, valid_peak_ids, self.output_directory, out_seq_filename)
