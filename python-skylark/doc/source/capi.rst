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

Initialization and Finalization
-------------------------------

Skylark is automatically initialized with a random seed when you import
cskylark. However, you can reinitialize it to a specific seed by calling
initialize. While not required, you can finalize the library using
finalize. However, note that that will not cause allocated objects
(e.g. sketch transforms) to be freed. They are freed by the garbage collector
when detected as garbage (no references).

.. autofunction:: initialize

.. autofunction:: finalize

SketchTransform
---------------

.. autoclass:: _SketchTransform
                :members:

                .. automethod:: _SketchTransform.__mul__

                .. automethod:: _SketchTransform.__div__

Supported Sketches
------------------

.. autoclass:: JLT
.. autoclass:: FJLT
.. autoclass:: SJLT
.. autoclass:: CT
.. autoclass:: CWT
.. autoclass:: MMT
.. autoclass:: WZT
.. autoclass:: GaussianRFT
.. autoclass:: LaplacianRFT
.. autoclass:: URST

Example
-------
.. literalinclude:: ../../skylark/examples/example_cskylark.py



