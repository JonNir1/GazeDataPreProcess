import unittest
import numpy as np

from Utils import angle_utils as au
from Utils.ScreenMonitor import ScreenMonitor


class TestUtils(unittest.TestCase):

    def test_calculate_visual_angle(self):
        # implausible values
        d = 1
        sm = ScreenMonitor(width=10, height=10, resolution=(10, 10), refresh_rate=60)
        self.assertEqual(45, au.calculate_visual_angle(d=d, p1=(0, 0), p2=(0, 1),
                                                       screen_monitor=sm, use_radians=False))
        self.assertEqual(45, au.calculate_visual_angle(d=d, p1=(0, 0), p2=(1, 0),
                                                       screen_monitor=sm, use_radians=False))
        self.assertAlmostEqual(70.528779366, au.calculate_visual_angle(d=d, p1=(0, 0), p2=(2, 2),
                                                                       screen_monitor=sm, use_radians=False))

        # plausible values
        d = 65
        sm = ScreenMonitor(width=53.5, height=31, resolution=(1920, 1080), refresh_rate=60)
        self.assertAlmostEqual(0.0247416922, au.calculate_visual_angle(d=d, p1=(0, 1), p2=(1, 1),
                                                                       screen_monitor=sm, use_radians=False))
        self.assertAlmostEqual(0.0247416922, au.calculate_visual_angle(d=d, p1=(1, 2), p2=(1, 1),
                                                                       screen_monitor=sm, use_radians=False))
        self.assertAlmostEqual(0.0349900345, au.calculate_visual_angle(d=d, p1=(0, 0), p2=(1, 1),
                                                                       screen_monitor=sm, use_radians=False))

    def test_calculate_visual_angle_velocities(self):
        # TODO
        pass
