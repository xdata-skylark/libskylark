import unittest
import random
import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, c_char_p, pointer, POINTER, c_bool
import ctypes.util
import sys

import skylark.lib as libpy



skylarklib = cdll.LoadLibrary('libcskylark.so')
skylarklib.sl_has_elemental.restype = c_bool
skylarklib.sl_has_combblas.restype  = c_bool


class libTestCase(unittest.TestCase):
    """Tests for `lib.py`."""

    def test_initialize_without_parameters(self):
        """Initialize lib with a seed provided by the system time."""
        libpy.initialize()
    
    def test_initialize_with_parameters(self):
        """Initialize lib with a random seed with values from 0 to 10000"""
        seed = int(random.random()*10000)
        libpy.initialize(seed)
    
    def test_finalize(self):
        """Finalize (de-allocate) the library"""
        libpy.finalize()

    def test_callsl(self):
        """Try to call a mockup function"""
        #TODO: call some function
        pass
        
    def test_map_to_constructor_basic(self):
        """Mapping between type string and and constructor exists for np and scipy"""
        keys = ["LocalMatrix", "LocalSpMatrix"]
        for i in keys:
            self.assertTrue(i in libpy.map_to_ctor)

    def test_map_to_constructor_El(self):
        """Mapping between type string and and constructor exists for El"""
        keys = ["ElMatrix", "DistMatrix", "DistMatrix_VR_STAR", "DistMatrix_VC_STAR", \
                 "DistMatrix_STAR_VR", "DistMatrix_STAR_VC", "SharedMatrix", \
                 "RootMatrix" \
               ]
        if skylarklib.sl_has_elemental():
            for i in keys:
                self.assertTrue(i in libpy.map_to_ctor)

    def test_map_to_constructor_KDT(self):
        """Mapping between type string and and constructor exists for KDT"""
        keys = ["DistSparseMatrix"]
        if skylarklib.sl_has_combblas():
            for i in keys:
                self.assertTrue(i in libpy.map_to_ctor)
        
    def test_all_map_to_cosntruct_tested(self):
        """There is no maps_to_construct untested"""
        # Basic Keys
        keys = ["LocalMatrix", "LocalSpMatrix"]
        
        # Add Elemental keys
        if skylarklib.sl_has_elemental():
            keys = keys + ["ElMatrix", "DistMatrix", "DistMatrix_VR_STAR", "DistMatrix_VC_STAR", \
                           "DistMatrix_STAR_VR", "DistMatrix_STAR_VC", "SharedMatrix", \
                           "RootMatrix" \
                          ]
        # Add KDT keys
        if skylarklib.sl_has_combblas():
            keys = keys +  ["DistSparseMatrix"]
        self.assertTrue(len(keys) == len(libpy.map_to_ctor))


class AdaptersTestCase(unittest.TestCase):
    """Tests for adapters in `lib.py`."""
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(libTestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)