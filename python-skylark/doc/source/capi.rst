Accessing the Lower Layers
==========================

.. currentmodule:: skylark.cskylark

Overview
--------

The skylark lower layers offer optimized distributed sketching transforms and
computational primitives. These can be accessed in Python through 
:mod:`skylark.cskylark`. At this point only the sketching transforms are 
accessible through the interface.

Dependencies
------------

Skylark uses external libraries to represent distributed matrices. For
dense distributed matrices it uses `Elemental <http://libelemental.org/>`_.
Currently it uses the c-types interface of Elemental, so be sure install
that as well. For sparse distributed matrices it uses 
`ComboBLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ interfaced
through `KDT <http://kdt.sourceforge.net/wiki/index.php/Main_Page>`_.

The lower layers use MPI so it is advisable an MPI interface to Python be 
installed. One option is to use `mpi4py <http://mpi4py.scipy.org/>`_.

The Context
-----------

.. autoclass:: Context
	       :members:

SketchTransform
---------------

.. autoclass:: SketchTransform
                :members:

Supported Sketches
------------------

.. autoclass:: JLT
.. autoclass:: FJLT
.. autoclass:: CT
.. autoclass:: CWT
.. autoclass:: MMT
.. autoclass:: WZT
.. autoclass:: GaussianRFT
.. autoclass:: LaplacianRFT

Example
-------
.. literalinclude:: ../../skylark/examples/example_cskylark.py   
   
   

