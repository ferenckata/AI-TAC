import unittest
import numpy as np
import src.data_processing.preprocess_utils as pp

class TestPreprocessingMethods(unittest.TestCase):

    def test_upper(self):
        test_string = "AAGCTG"
        expected_encoding = np.array([1,1,0,0,0,0,
                                      0,0,0,0,1,0,
                                      0,0,1,0,0,1,
                                      0,0,0,1,0,0], dtype = 'int8').reshape(4,6)
        actual_encoding = pp.one_hot_encoder(test_string)
        np.testing.assert_array_equal(actual_encoding, expected_encoding)

    def test_lower(self):
        test_string = "aagctg"
        expected_encoding = np.array([1,1,0,0,0,0,
                                      0,0,0,0,1,0,
                                      0,0,1,0,0,1,
                                      0,0,0,1,0,0], dtype = 'int8').reshape(4,6)
        actual_encoding = pp.one_hot_encoder(test_string)
        np.testing.assert_array_equal(actual_encoding, expected_encoding)

    def test_mixed(self):
        test_string = "aAgCTg"
        expected_encoding = np.array([1,1,0,0,0,0,
                                      0,0,0,0,1,0,
                                      0,0,1,0,0,1,
                                      0,0,0,1,0,0], dtype = 'int8').reshape(4,6)
        actual_encoding = pp.one_hot_encoder(test_string)
        np.testing.assert_array_equal(actual_encoding, expected_encoding)

    def test_N(self):
        test_string = "aAnCNg"
        expected_encoding = "contains_N"
        actual_encoding = pp.one_hot_encoder(test_string)
        self.assertEqual(actual_encoding, expected_encoding)

    def test_nonvalid_character(self):
        test_string = "aAnCXg"
        expected_encoding = "contains_N"
        actual_encoding = pp.one_hot_encoder(test_string)
        self.assertEqual(actual_encoding, expected_encoding)
