"""File to provide entry points to user interaction"""

from data_processing.preprocess_data import Preprocessor
from IO.IO import IO

class UserInterface():
    """
    Class to provide entry points to user interaction
    """

    def run_preprocessing(self):
        """
        Method to run preprocessing steps
        """
        my_io = IO("config.yml")
        data_file = my_io.get_config_entry("data_file")
        intensity_file = my_io.get_config_entry("intensity_file")
        reference_genome_dir = my_io.get_config_entry("reference_genome_dir")
        data_type = my_io.get_config_entry("data_type")
        num_chr = my_io.get_config_entry("num_chr")
        region_size = my_io.get_config_entry("region_size")
        my_preprocessor = Preprocessor(data_file, intensity_file, reference_genome_dir, data_type, num_chr, region_size)
        positions, chr_dict = my_preprocessor.read_in_data()
        valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids = my_preprocessor.reformat_data(positions, chr_dict)
        my_preprocessor.save_data_to_file(valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids)
