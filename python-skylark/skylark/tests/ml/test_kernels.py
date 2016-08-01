import unittest
import numpy as np

import skylark.ml.kernels as sl_ml_kernels
from .. import utils




class LinearTestCase(unittest.TestCase):
    """Tests linear kernel over d dimensional vectors."""

    def test_Linear_gram(self):
        """Returns the dense Gram matrix evaluated over the datapoints."""
        pass
    
    def test_Linear_rft(self):
        """Create random features transform for the kernel."""
        pass
      
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DistancesTestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)
