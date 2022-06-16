"""
Collection of used methods when preprocessing the data
for subsequent deep learning application
"""
from collections import defaultdict
from Bio import SeqIO
import numpy as np

class PreprocessingMethods():
    """
    A class used to provide general methods used when reading in an reformating
    data for deep learning application
    ...

    Static Methods
    -------
    read_bed(filename):
        Read names and postions from bed file

    read_fasta(genome_dir, num_chr):
        Parse fasta file and turn into dictionary, save into file if not saved yet
    
    read_intensities(intensity_file):
        Read in peak values from tab delimited file

    one_hot_encoder(sequence):
        Encodes a DNA sequence to one-hot-encoded matrix with rows A, T, G, C

    get_sequences(positions, chr_dict, num_chr):
        Get sequences for peaks from the reference genome
    
    filter_matrix(names, valid_ids, *argv):
        Filter out invalid entries from each data matrix

    """

    @staticmethod
    def read_bed(filename):
        """
        Read names and postions from bed file

        Parameters
        ----------
        filename: str
            Path to the location of the input bed file with peak coordinates
            The bed file should have four columns in this order: chr, start, end, name
        
        Return
        ------
        positions: dict
            Dictionary with keys: location IDs, and values: tuple of (chr, start, end)
        """
        positions = defaultdict(list)
        with open(filename, encoding='utf8') as bed_file:
            for line in bed_file:
                chromosome, start, stop, name = line.split()
                positions[name].append((chromosome, int(start), int(stop)))

        return positions


    @staticmethod
    def read_fasta(genome_dir, num_chr):
        """
        Parse fasta file and turn into dictionary, save into file if not saved yet

        Parameters
        ----------
        genome_dir: str
            Path to the directory with reference genome fasta files
        num_chr: int
            Number of chromosomes, this should match the name of the reference genome
            fasta file names in the following manner: chr{num_chr}.fa
        Return
        ------
        chr_dict: dict
            Dictionary of genomic sequences (one entry per fasta entry (= chromosome))
        """
        chr_dict = {}
        for chromosome in range(1, num_chr):
            # what happens when the file does not exist?
            chr_file_path = genome_dir + f'chr{chromosome}.fa'
            # in case memory becomes an issue, use Bio.SeqIO.index() instead
            chr_dict.update(SeqIO.to_dict(SeqIO.parse(chr_file_path, 'fasta')))

        return chr_dict


    @staticmethod
    def read_intensities(intensity_file):
        """
        Read in peak values from tab delimited file

        Parameters
        ----------
        intensity_file: str
            Path to the tab (or whitespace) deliminated text file with normalized peak heights

        Return
        ------
        cell_type_array: numpy.ndarray
            Matrix of peak intensity values across cell type (num_positions x num_celltypes)
        peak_names: numpy.ndarray
            Array of peak names (num_peak_name x num_celltypes)
        """
        cell_type_array = []
        peak_names = []
        with open(intensity_file, encoding='utf8') as peak_file:
            for i, line in enumerate(peak_file):
                # skip first line of IDs
                if i == 0:
                    continue
                columns = line.split()
                peak_name = columns[0]
                # read lines until the EOF is read
                if '\x1a' not in columns:
                    cell_act = columns[1:] # removes peak ID
                    cell_type_array.append(cell_act)
                    peak_names.append(peak_name)

        cell_type_array = np.stack(cell_type_array)
        peak_names = np.stack(peak_names)

        return cell_type_array, peak_names


    @staticmethod
    def one_hot_encoder(sequence):
        """
        Encodes a DNA sequence to one-hot-encoded matrix with rows A, T, G, C

        Parameters
        ----------
        sequence: str
            Input sequence to encode

        Return
        ------
        encoding: numpy.ndarray or None
            One-hot encoded sequence matrix if all characters are valid (A,T,G,C) (4 x seq_length)
            None otherwise
        """
        seq_len = len(sequence)
        encoding = np.zeros((4,seq_len),dtype = 'int8')
        for j, i in enumerate(sequence):
            if i == "a":
                encoding[0][j] = 1
            elif i == "t":
                encoding[1][j] = 1
            elif i == "g":
                encoding[2][j] = 1
            elif i == "c":
                encoding[3][j] = 1
            else:
                return None

        return encoding

    @staticmethod
    def get_sequences(positions, chr_dict, num_chr):
        """
        Get sequences for peaks from the reference genome

        Parameters
        ----------
        positions: dict
            Dictionary with keys: location IDs, and values: tuple of (chr, start, end)
        chr_dict: dict
            Dictionary of genomic sequences (one entry per fasta entry (= chromosome))
        num_chr: int
            Number of chromosomes, this should match the name of the reference genome
            fasta file names in the following manner: chr{num_chr}.fa

        Return
        ------
        one_hot_seqs: numpy.ndarray
            One-hot encoded sequences of input regions (num_positions x 4 x seq_length)
        peak_seqs: numpy.ndarray
            Sequences of input regions (num_positions x seq_length ?? )
        peak_names: numpy.ndarray
            Names of valid input regions (num_valid_positions x name_length ?? )
        invalid_ids: list
            Names of invalid input regions (num_invalid_positions x name_length ?? )
        """
        one_hot_seqs = []
        peak_seqs = []
        invalid_ids = []
        peak_names = []

        target_chr = [f'chr{i}' for i in range(1, num_chr)]

        for name in positions:
            for (chromosome, start, stop) in positions[name]:
                # somewhat unnecessary here as it would probably throw an error
                # when reading the file in
                # todo: check it once above
                if chromosome in target_chr:
                    chr_seq = chr_dict[chromosome].seq
                    peak_seq = str(chr_seq)[start - 1:stop].lower()
                    # already lowered the character, no need to check for uppercase
                    # letter in the encoding step
                    one_hot_seq = PreprocessingMethods.one_hot_encoder(peak_seq)
                    # check if it is valid sequence
                    if isinstance(one_hot_seq, np.ndarray):
                        peak_names.append(name)
                        peak_seqs.append(peak_seq)
                        one_hot_seqs.append(one_hot_seq)
                    else:
                        invalid_ids.append(name)
                else:
                    invalid_ids.append(name)

        one_hot_seqs = np.stack(one_hot_seqs)
        peak_seqs = np.stack(peak_seqs)
        peak_names = np.stack(peak_names)

        return one_hot_seqs, peak_seqs, peak_names, invalid_ids


    @staticmethod
    def filter_matrix(names, valid_ids, *argv):
        """
        Filter out invalid entries from each data matrix

        Parameters
        ----------
        names: numpy.ndarray
            Names of inputs (num_names x name_length ?? )
        valid_ids: numpy.ndarray
            Mask of indeces of valid inputs (1 x num_names)

        Return
        ------
        return_values: list of numpy.ndarrays
            Names of valid inputs (num_valid_names x name_length ?? )
            Other arrays of valid inputs

        """
        valid_name_mask = np.isin(names, valid_ids)
        return_values = []
        return_values.append(names[valid_name_mask])
        for arg in argv:
            return_values.append(arg[valid_name_mask, ...])
  
        return return_values
