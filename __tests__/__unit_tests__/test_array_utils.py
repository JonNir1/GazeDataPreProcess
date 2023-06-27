import unittest
import numpy as np

from Utils import array_utils as au


class TestUtils(unittest.TestCase):

    def test_normalize_array_1d(self):
        arr = np.arange(8).astype(float)
        arr[0] = np.nan
        arr[3] = np.nan
        normalized = au.normalize_array(arr)
        expected = np.array([np.nan, 0, 1/6, np.nan, 3/6, 4/6, 5/6, 1])
        for i in range(len(arr)):
            if np.isnan(expected[i]):
                self.assertTrue(np.isnan(normalized[i]))
            else:
                self.assertEqual(expected[i], normalized[i])

    def test_normalize_array_2d(self):
        arr = np.arange(4).astype(float)
        arr[1] = np.nan
        arr_2d = arr[:, np.newaxis] * arr
        normalized_2d = au.normalize_array(arr_2d)
        expected = np.array([[0, np.nan, 0, 0],
                             [np.nan, np.nan, np.nan, np.nan],
                             [0, np.nan, 4 / 9, 6 / 9],
                             [0, np.nan, 6 / 9, 9 / 9]])
        for i in range(len(arr)):
            for j in range(len(arr)):
                if np.isnan(expected[i, j]):
                    self.assertTrue(np.isnan(normalized_2d[i, j]))
                else:
                    self.assertEqual(expected[i, j], normalized_2d[i, j])

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
        # TODO
        pass

    def test_median_standard_deviation(self):
        # TODO
        pass

    def test_get_different_event_indices(self):
        # TODO
        pass



