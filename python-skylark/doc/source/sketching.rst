Sketching Transforms
=====================
 
Just some nomenclature:

A **left sketch** of a matrix **A** is a matrix of the form **B = S A** where *S* is a sketch operator. 
In other words, a left sketch performs dimensionality reduction of the columns of *A*, and reduces the number of rows of A.

Likewise, a **right sketch** is of the form **A S**. It reduces dimensionality of row-vectors and shrinks the number of columns. 

By a **two-sided sketch**, we mean both left and right sketching **S A R**.
  
.. currentmodule:: skylark.sketch

Johnson-Lindenstrauss (JL) with Dense (Gaussian) Sketching
-------------------------------------------------------

.. autoclass:: skylark.sketch.JL
	:members:

Faster Johnson-Lindenstrauss (FJLT) 
------------------------------------

.. autoclass:: skylark.sketch.FJLT
	:members:

Sparse Sketching Transforms
-----------------------------------

.. autoclass:: skylark.sketch.SparseJL
	:members:

