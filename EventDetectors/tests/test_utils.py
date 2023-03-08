import unittest
import numpy as np

import EventDetectors.utils as u


class TestUtils(unittest.TestCase):

    # TODO: test for correct exceptions as well

    @staticmethod
    def test_numeric_derivative():
        arr1 = np.arange(10)
        n = 3
        arr1_deriv = u.numerical_derivative(arr1, n)
        np.testing.assert_equal(arr1_deriv, np.concatenate([[np.nan]*(n-1), [1]*6, [np.nan]*(n-1)]))
        n = 4
        arr1_deriv = u.numerical_derivative(arr1, n)
        np.testing.assert_equal(arr1_deriv, np.concatenate([[np.nan] * (n - 1), [1.5] * 4, [np.nan] * (n - 1)]))

    def test_median_standard_deviation(self):
        arr1 = np.arange(10)
        calculated = u.median_standard_deviation(arr1)
        expected = 0.5
        self.assertEqual(calculated, expected)


