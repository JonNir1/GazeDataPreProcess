import unittest
import numpy as np

import visual_angle_utils as vau


class TestUtils(unittest.TestCase):

    def test_pixels2deg(self):
        # implausible values
        dist = 1
        screen_size = (10, 10)
        screen_res = (10, 10)
        self.assertEqual(45, vau.pixels2deg(np.array([[0, 0], [0, 1]]), dist, screen_size, screen_res))
        self.assertEqual(45, vau.pixels2deg(np.array([[0, 0], [1, 0]]), dist, screen_size, screen_res))
        self.assertAlmostEqual(70.528779366, vau.pixels2deg(np.array([[0, 0], [2, 2]]), dist, screen_size, screen_res))

        # plausible values
        dist = 65
        screen_size = (53.5, 31)
        screen_res = (1920, 1080)
        self.assertAlmostEqual(0.0245618907, vau.pixels2deg(np.array([[0, 1], [1, 1]]), dist, screen_size, screen_res))
        self.assertAlmostEqual(0.0253015533, vau.pixels2deg(np.array([[1, 2], [1, 1]]), dist, screen_size, screen_res))
        self.assertAlmostEqual(0.0352626561, vau.pixels2deg(np.array([[0, 0], [1, 1]]), dist, screen_size, screen_res))

    def test_rad2deg(self):
        self.assertEqual(180, vau.rad2deg(np.pi))
        self.assertEqual(45, vau.rad2deg(np.pi / 4))
        self.assertEqual(135, vau.rad2deg(3 * np.pi / 4))

    def test_deg2rad(self):
        self.assertEqual(np.pi, vau.deg2rad(180))
        self.assertEqual(np.pi / 4, vau.deg2rad(45))
        self.assertEqual(3 * np.pi / 4, vau.deg2rad(135))

    def test_pixel_width_degrees(self):
        self.assertEqual(45, vau.pixel_width_degrees(distance=1, screen_width=1, screen_resolution=1))
        self.assertEqual(45, vau.pixel_width_degrees(distance=1, screen_width=10, screen_resolution=10))
        self.assertAlmostEqual(0.02456189,
                               vau.pixel_width_degrees(distance=65, screen_width=53.5, screen_resolution=1920))

    def test_pixel_height_degrees(self):
        self.assertEqual(45, vau.pixel_height_degrees(distance=1, screen_height=1, screen_resolution=1))
        self.assertEqual(45, vau.pixel_height_degrees(distance=1, screen_height=10, screen_resolution=10))
        self.assertAlmostEqual(0.02530155,
                               vau.pixel_height_degrees(distance=65, screen_height=31, screen_resolution=1080))
