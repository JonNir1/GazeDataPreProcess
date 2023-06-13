import unittest
import numpy as np

from Utils import array_utils as au


class TestUtils(unittest.TestCase):

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



