import unittest
import numpy as np

from Utils import array_utils as au


class TestUtils(unittest.TestCase):

    def test_normalize_array_1d(self):
        arr = np.arange(8).astype(float)
        arr[0] = np.nan
        arr[3] = np.nan
        normalized = au.normalize_array(arr)
        expected = np.array([np.nan, 0, 1 / 6, np.nan, 3 / 6, 4 / 6, 5 / 6, 1])
        self.assertTrue(np.array_equal(expected, normalized, equal_nan=True))

    def test_normalize_array_2d(self):
        arr = np.arange(4).astype(float)
        arr[1] = np.nan
        arr_2d = arr[:, np.newaxis] * arr
        normalized_2d = au.normalize_array(arr_2d)
        expected = np.array([[0, np.nan, 0, 0],
                             [np.nan, np.nan, np.nan, np.nan],
                             [0, np.nan, 4 / 9, 6 / 9],
                             [0, np.nan, 6 / 9, 9 / 9]])
        self.assertTrue(np.array_equal(expected, normalized_2d, equal_nan=True))

    def test_shift_array(self):
        arr = np.arange(8)
        n = 1

        shifted_plus = au.shift_array(arr, 1)
        for i in range(n):
            self.assertTrue(np.isnan(shifted_plus[i]))
        for i in range(n, len(arr)):
            self.assertEqual(arr[i - 1], shifted_plus[i])

        shifted_minus = au.shift_array(arr, -2)
        for i in range(len(arr) - n):
            self.assertEqual(arr[i + n], shifted_minus[i])
        for i in range(len(arr) - n, len(arr)):
            self.assertTrue(np.isnan(shifted_minus[i]))

    def test_numerical_derivative(self):
        arr = np.ones(10)
        self.assertRaises(ValueError, au.numerical_derivative, arr, 0)
        self.assertRaises(ValueError, au.numerical_derivative, arr, 5)
        res = au.numerical_derivative(arr, n=2)
        expected = np.array([np.nan, 0, 0, 0, 0, 0, 0, 0, 0, np.nan])
        np.array_equal(expected, res, equal_nan=True)

        arr = np.arange(10)
        res = au.numerical_derivative(arr, n=1)
        expected = np.array([np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan])
        np.array_equal(expected, res, equal_nan=True)

    def test_median_standard_deviation(self):
        # TODO
        pass

    def test_get_chunk_indices(self):
        arr = np.hstack([np.zeros(3, dtype=bool), np.ones(4, dtype=bool),
                         np.zeros(3, dtype=bool), np.ones(2, dtype=bool)])
        res0 = au.get_chunk_indices(arr, min_length=0)
        expected0 = [list(range(3, 7)), list(range(10, 12))]
        self.assertEqual(len(res0), len(expected0))
        for i in range(len(res0)):
            self.assertTrue(np.array_equal(res0[i], expected0[i]))

        res3 = au.get_chunk_indices(arr, min_length=3)
        expected3 = [list(range(3, 7))]
        self.assertEqual(len(res3), len(expected3))
        for i in range(len(res3)):
            self.assertTrue(np.array_equal(res3[i], expected3[i]))
