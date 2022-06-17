""" Test file for preprocessing methods """
from collections import defaultdict
import unittest
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import src.data_processing.preprocess_utils as pp

class TestPreprocessingMethods(unittest.TestCase):
    """
    Testing class for preprocessing methods
    """

    def test_bedfile_reader_standard_file(self):
        """
        Test code to read in bedfile with chr, start, stop, name entries separated by tab (equal entry length)
        """
        test_bed = "test/correct_bed_testfile.bed"
        region_size = 250
        expected_output = defaultdict(list)
        expected_output["peak_1"].append(("chr1",50,300))
        expected_output["peak_2"].append(("chr2",200,450))
        expected_output["peak_3"].append(("chr3",150,400))
        expected_skipped = []
        actual_output, actual_skipped = pp.PreprocessingMethods.read_bed(test_bed, region_size)
        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_skipped, expected_skipped)


    def test_bedfile_reader_nonfixed_length_file(self):
        """
        Test code to read in bedfile with unequal entry length
        """
        test_bed = "test/incorrect_bed_testfile.bed"
        region_size = 250
        expected_positions = defaultdict(list)
        expected_skipped = ["peak_1", "peak_2", "peak_3"]
        actual_positions, actual_skipped = pp.PreprocessingMethods.read_bed(test_bed, region_size)
        self.assertEqual(actual_positions, expected_positions)
        self.assertEqual(actual_skipped, expected_skipped)


    def test_fastafile_reader_standard_file(self):
        """
        Test code to read in fasta files from directory
        """
        test_fasta_dir = "test/fasta_folder"
        test_chr_num = 2
        record_test1 = SeqRecord(
            Seq("aaaGTTNCNGT"),
            id="test1",name="test1",
            description="test1",)
        record_test2 = SeqRecord(
            Seq("AAGCCTTGCGCAGC"),
            id="test2",name="test2",
            description="test2",)
        record_test3 = SeqRecord(
            Seq("ccgtgctgctgactga"),
            id="test3",name="test3",
            description="test3",)
        record_test4 = SeqRecord(
            Seq("gtgatgatgatag"),
            id="test4",name="test4",
            description="test4",)
        record_test5 = SeqRecord(
            Seq("fegfevFTDF"),
            id="test5",name="test5",
            description="test5",)
        record_test6 = SeqRecord(
            Seq("AGATGATAGATAGN"),
            id="test6",name="test6",
            description="test6",)
        record_test7 = SeqRecord(
            Seq("AGATAGTAGAAATGGCGCGC"),
            id="test7",name="test7",
            description="test7",)
        record_testX = SeqRecord(
            Seq("agatgctgctgatcgatcgatgctaggatcga"),
            id="testX",name="testX",
            description="testX",)
        expected_output = {"test1":record_test1,
                           "test2":record_test2,
                           "test3":record_test3,
                           "test4":record_test4,
                           "test5":record_test5,
                           "test6":record_test6,
                           "test7":record_test7,
                           "testX":record_testX}
        actual_output = pp.PreprocessingMethods.read_fasta(test_fasta_dir, test_chr_num)
        self.assertEqual([rec.seq for rec in actual_output.values()],
                        [rec.seq for rec in expected_output.values()])
        self.assertEqual(list(actual_output), list(expected_output))


    def test_fastafile_reader_xy_chr(self):
        """
        Test code to read in fasta files including X and Y chromosomes from directory
        """
        # ToDo


    def test_intensityfile_reader_standard_file(self):
        """
        Test code to read in tab delimited file of peak value - celltype matrix
        """
        test_tsv = "test/intensity_testfile.tsv"
        expected_cell_type_array = np.stack([['0.41', '0.71', '0.9','0.11'],
                                            ['0.41', '1.64', '0.9', '0.83'],
                                            ['2.36', '0.1', '0.9', '0.11']])
        expected_peak_names = np.stack(["peak_1", "peak_2", "peak_3"])
        actual_cell_type_array, actual_peak_names = pp.PreprocessingMethods.read_intensities(test_tsv)
        np.testing.assert_array_equal(actual_cell_type_array, expected_cell_type_array)
        np.testing.assert_array_equal(actual_peak_names, expected_peak_names)


    def test_encoding_lower(self):
        """
        Test encoding of lowercase sequence characters
        """
        test_string = "aagctg"
        expected_encoding = np.array([1,1,0,0,0,0,
                                      0,0,0,0,1,0,
                                      0,0,1,0,0,1,
                                      0,0,0,1,0,0], dtype = 'int8').reshape(4,6)
        actual_encoding = pp.PreprocessingMethods.one_hot_encoder(test_string)
        np.testing.assert_array_equal(actual_encoding, expected_encoding)


    def test_encoding_n(self):
        """
        Test encoding of sequence characters including n
        """
        test_string = "aancng"
        expected_encoding = None
        actual_encoding = pp.PreprocessingMethods.one_hot_encoder(test_string)
        self.assertEqual(actual_encoding, expected_encoding)


    def test_encoding_nonvalid_character(self):
        """
        Test encoding of sequence characters including characters not a,c,t,g,n
        """
        test_string = "aan cxg/?"
        expected_encoding = None
        actual_encoding = pp.PreprocessingMethods.one_hot_encoder(test_string)
        self.assertEqual(actual_encoding, expected_encoding)

    
    def test_get_sequences(self):
        """
        Test the method which loops through the sequences and encodes them
        """
        test_positions = defaultdict(list)
        test_positions["peak_1"].append(("chr1",1,5))
        test_positions["peak_2"].append(("chr2",6,10))
        test_positions["peak_3"].append(("chr2",23,27))
        
        record_test1 = SeqRecord(
            Seq("AGNTGATAGATAGAGTGTATGTA"),
            id="chr1",name="chr1",
            description="chr1",)
        record_test2 = SeqRecord(
            Seq("AGATAGTAGAAATGGCGCGCTGGCGCGCAATATAGTAATTGGAA"),
            id="chr2",name="chr2",
            description="chr2",)
        test_chr_seq = {"chr1":record_test1,
                        "chr2":record_test2
                        }
        test_num_chr = 2
        expected_ohs = np.stack((np.array([0,0,1,0,1,
                                           0,1,0,0,0,
                                           1,0,0,1,0,
                                           0,0,0,0,0], dtype = 'int8').reshape(4,5),
                                np.array([0,0,0,0,0,
                                          0,0,0,0,0,
                                          1,0,1,0,1,
                                          0,1,0,1,0], dtype = 'int8').reshape(4,5)
                                ))
        expected_ps = np.stack(["GTAGA".lower(), "GCGCG".lower()])
        expected_pn = np.stack(["peak_2","peak_3"])
        ecpected_ii = ["peak_1"]
        actual_ohs, actual_ps, actual_pn, actual_ii = pp.PreprocessingMethods.get_sequences(test_positions, test_chr_seq, test_num_chr)
        np.testing.assert_array_equal(actual_ohs, expected_ohs)
        np.testing.assert_array_equal(actual_ps, expected_ps)
        np.testing.assert_array_equal(actual_pn, expected_pn)
        self.assertEqual(actual_ii, ecpected_ii)


