import unittest
import sys

sys.path.append("../")
from flow.build_circuits import *


class TestQC(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        pass

    def tearDown(self):
        # Cleanup code to run after each test
        pass

    def test_example_1(self):
        # Example test case 1
        self.assertEqual(1 + 1, 2)

    def test_example_2(self):
        # Example test case 2
        self.assertTrue(isinstance("hello", str))

    def test_example_3(self):
        # Example test case 3
        self.assertFalse(3 > 5)


if __name__ == "__main__":
    unittest.main()
