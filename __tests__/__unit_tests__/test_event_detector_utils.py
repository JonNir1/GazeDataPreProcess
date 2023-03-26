import unittest
import numpy as np

import EventDetectors.scripts.event_detector_utils as eu


class TestUtils(unittest.TestCase):

    def test_shift_array(self):
        arr = np.arange(8)
        n = 1

        shifted_plus = eu.shift_array(arr, 1)
        for i in range(n):
            self.assertTrue(np.isnan(shifted_plus[i]))
        for i in range(n, len(arr)):
            self.assertEqual(arr[i - 1], shifted_plus[i])

        shifted_minus = eu.shift_array(arr, -2)
        for i in range(len(arr) - n):
            self.assertEqual(arr[i + n], shifted_minus[i])
        for i in range(len(arr) - n, len(arr)):
            self.assertTrue(np.isnan(shifted_minus[i]))



