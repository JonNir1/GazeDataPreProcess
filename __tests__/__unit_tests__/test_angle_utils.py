import unittest
import numpy as np

from Utils import angle_utils as angle_utils
from Config.ScreenMonitor import ScreenMonitor


class TestUtils(unittest.TestCase):

    def test_calculate_azimuth(self):
        # angles are counter-clockwise from the positive x-axis, with y-axis pointing down
        self.assertEqual(0, angle_utils.calculate_azimuth(p1=(0, 0), p2=(0, 0), use_radians=False))
        self.assertEqual(45, angle_utils.calculate_azimuth(p1=(0, 0), p2=(1, -1), use_radians=False))
        self.assertEqual(315, angle_utils.calculate_azimuth(p1=(0, 0), p2=(1, 1), use_radians=False))
        self.assertEqual(90, angle_utils.calculate_azimuth(p1=(1, 1), p2=(1, -1), use_radians=False))
        self.assertEqual(180, angle_utils.calculate_azimuth(p1=(1, 1), p2=(1, -1),
                                                            use_radians=False, zero_direction='S'))

        self.assertEqual(np.pi * 3 / 4, angle_utils.calculate_azimuth(p1=(1, 1), p2=(-2, -2), use_radians=True))
        self.assertEqual(np.pi, angle_utils.calculate_azimuth(p1=(1, 0), p2=(-1, 0), use_radians=True))
        self.assertEqual(np.pi * 5 / 4, angle_utils.calculate_azimuth(p1=(1, 0), p2=(0, 1), use_radians=True))
        self.assertEqual(np.pi * 3 / 4, angle_utils.calculate_azimuth(p1=(1, 0), p2=(0, 1),
                                                                      use_radians=True, zero_direction='N'))

    def test_calculate_visual_angle(self):
        # implausible values
        d = 1
        sm = ScreenMonitor(width=10, height=10, resolution=(10, 10), refresh_rate=60)
        self.assertEqual(45, angle_utils.calculate_visual_angle(p1=(0, 0), p2=(0, 1), d=d, pixel_size=sm.pixel_size,
                                                                use_radians=False))
        self.assertEqual(45, angle_utils.calculate_visual_angle(p1=(0, 0), p2=(1, 0), d=d, pixel_size=sm.pixel_size,
                                                                use_radians=False))
        self.assertAlmostEqual(70.528779366,
                               angle_utils.calculate_visual_angle(p1=(0, 0), p2=(2, 2), d=d, pixel_size=sm.pixel_size,
                                                                  use_radians=False))

        # plausible values
        d = 65
        sm = ScreenMonitor(width=53.5, height=31, resolution=(1920, 1080), refresh_rate=60)
        self.assertAlmostEqual(0.0247416922,
                               angle_utils.calculate_visual_angle(p1=(0, 1), p2=(1, 1), d=d, pixel_size=sm.pixel_size,
                                                                  use_radians=False))
        self.assertAlmostEqual(0.0247416922,
                               angle_utils.calculate_visual_angle(p1=(1, 2), p2=(1, 1), d=d, pixel_size=sm.pixel_size,
                                                                  use_radians=False))
        self.assertAlmostEqual(0.0349900345,
                               angle_utils.calculate_visual_angle(p1=(0, 0), p2=(1, 1), d=d, pixel_size=sm.pixel_size,
                                                                  use_radians=False))

    def test_calculate_visual_angle_velocities(self):
        # TODO
        pass

    def test_calculate_pixels_from_visual_angle(self):
        # implausible values
        d = 1
        sm = ScreenMonitor(width=10, height=10, resolution=(10, 10), refresh_rate=60)
        self.assertEqual(0, angle_utils.visual_angle_to_pixels(d=d, angle=0, pixel_size=sm.pixel_size))
        self.assertAlmostEqual(2, angle_utils.visual_angle_to_pixels(d=d, angle=90, pixel_size=sm.pixel_size))
        self.assertAlmostEqual(0.0261814,
                               angle_utils.visual_angle_to_pixels(d=d, angle=1.5, pixel_size=sm.pixel_size))

        # plausible values
        d = 65
        sm = ScreenMonitor(width=53.5, height=31, resolution=(1920, 1080), refresh_rate=60)
        self.assertEqual(0, angle_utils.visual_angle_to_pixels(d=d, angle=0, pixel_size=sm.pixel_size))
        self.assertAlmostEqual(60.62987,
                               angle_utils.visual_angle_to_pixels(d=d, angle=1.5, pixel_size=sm.pixel_size))
