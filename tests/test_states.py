import unittest
import sys

sys.path.append("../")

from QSD.states import NonOrthogonalStates
from QSD.flow.problem_spec import *


class TestNonOrthogonalStates(unittest.TestCase):

    def test_initial_state(self):
        state = NonOrthogonalStates()
        self.assertIsInstance(state, NonOrthogonalStates)

    def test_noise_channel(self):
        import numpy as np

        print()
        print(depolarizing_noise_channel(3))
        self.assertTrue(np.trace(depolarizing_noise_channel(3)) == 1)


if __name__ == "__main__":
    unittest.main()
