"""File to provide entry points to user interaction"""

from data_processing.preprocess_data import Preprocessor
from src.IO import IO

class UserInterface():
    """
    Class to provide entry points to user interaction
    """

    def run_preprocessing(self):
        """
        Method to run preprocessing steps
        """
        my_io = IO("/projects/ec31/mathelier-group/katalitf/AITAC_test/AI-TAC_refactored/AI-TAC/config.yml")
        data_file = my_io.get_config_entry("data_file")
        intensity_file = my_io.get_config_entry("intensity_file")
        reference_genome_dir = my_io.get_config_entry("reference_genome_dir")
        output_directory = my_io.get_config_entry("output_directory")
        num_chr = my_io.get_config_entry("num_chr")
        region_size = my_io.get_config_entry("region_size")
        my_preprocessor = Preprocessor(data_file, intensity_file, reference_genome_dir, output_directory, num_chr, region_size)
        my_io.log_to_file("initialized Preprocessor")
        positions, chr_dict, skipped_peaks = my_preprocessor.read_in_data()
        my_io.log_to_file("read data")
        valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids = my_preprocessor.reformat_data(positions, chr_dict)
        my_io.log_to_file("reformat data")
        my_preprocessor.save_data_to_file(valid_peak_ids, one_hot_seqs, peak_seqs, cell_type_array, invalid_ids, skipped_peaks)
        my_io.log_to_file("saved to file")
        my_io.log_to_file("done")

