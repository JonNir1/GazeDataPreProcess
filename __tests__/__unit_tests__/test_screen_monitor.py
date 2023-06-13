import unittest
from Utils.ScreenMonitor import ScreenMonitor


class TestUtils(unittest.TestCase):

    def test__calc_visual_angle_radius(self):
        # implausible values
        d = 1
        sm = ScreenMonitor(width=10, height=10, resolution=(10, 10), refresh_rate=60)
        self.assertEqual(0, sm.calc_visual_angle_radius(d=d, angle=0))
        self.assertAlmostEqual(1, sm.calc_visual_angle_radius(d=d, angle=90))
        self.assertAlmostEqual(0.0130907171, sm.calc_visual_angle_radius(d=d, angle=1.5))

        # plausible values
        d = 65
        sm = ScreenMonitor(width=53.5, height=31, resolution=(1920, 1080), refresh_rate=60)
        self.assertEqual(0, sm.calc_visual_angle_radius(d=d, angle=0))
        self.assertAlmostEqual(2315.75816791, sm.calc_visual_angle_radius(d=d, angle=90))
        self.assertAlmostEqual(30.314935013, sm.calc_visual_angle_radius(d=d, angle=1.5))
