import unittest
from unittest.mock import mock_open, patch
import numpy as np

from Gaussian_Band import read_wavelengths_from_hdr

class TestReadWavelengthsFromHdr(unittest.TestCase):
    def test_valid_hdr_file(self):
        mock_hdr_content = """
        some header info
        WAVELENGTHS
        450.0
        550.0
        650.0
        WAVELENGTHS_END
        """

        with patch("builtins.open", mock_open(read_data=mock_hdr_content)), \
             patch("os.path.exists", return_value=True):
            result = read_wavelengths_from_hdr("dummy.hdr")
            expected = np.array([450.0, 550.0, 650.0])
            np.testing.assert_array_almost_equal(result, expected)

    def test_missing_hdr_file(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                read_wavelengths_from_hdr("missing.hdr")

    def test_invalid_format(self):
        mock_hdr_content = """
        WAVELENGTHS
        abc
        123.4
        WAVELENGTHS_END
        """

        with patch("builtins.open", mock_open(read_data=mock_hdr_content)), \
             patch("os.path.exists", return_value=True):
            result = read_wavelengths_from_hdr("dummy.hdr")
            expected = np.array([123.4])
            np.testing.assert_array_almost_equal(result, expected)

    def test_no_wavelengths_found(self):
        mock_hdr_content = """
        WAVELENGTHS
        abc
        def
        WAVELENGTHS_END
        """

        with patch("builtins.open", mock_open(read_data=mock_hdr_content)), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                read_wavelengths_from_hdr("dummy.hdr")

if __name__ == "__main__":
    unittest.main()