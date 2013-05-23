Meeting Notes
==============

Monday, November 26, 2012
--------------------------
*Discussion of parallel pseudo-random number generators: how to get reproducible outputs independent of processor architecture.*

The context of the discussion is: consider parallel matrix multiplication, A*B, where A is an implicitly defined Gaussian 
Random matrix. Partition A and B into blocks of conforming size. Each block of A can be associated with a random seed. 
Each parallel task involving a block of A needs to be aware of the seed so that it can generate the same random stream. 
One advantage of this scheme is that no communication happens between nodes. A disadvantage is that the random matrix
if block dependent. Since optimal block size is dependent on the underlying parallel architecture, this makes our 
random mapping, a mathematical operator, a function of the underlying parallel platform. Ideally, irrespective of block sizes
and number of processors, given a "global seed", we would like the same mathematical operation to be performed, for the 
sake of reproducibilty.  


*Observation* 

For Random sampling, FFT and Hashing-based sketching, the state of the sketch is small. 
The state can be materialized sequentially and then communicated to all the processors, bypassing the PPRNG complication above, 
whcih arises for JL sketches based on Gaussian/Sign matrices.

*Followup:* 
 
SPRNG -- MPI-package for PPRNG: http://sprng.cs.fsu.edu/

PPRNG article: http://sprng.cs.fsu.edu/Version1.0/paper/index.html

Monday, December 4, 2012
---------------------------

Common data format, matrix data structures for the following three pieces of code:

#. Random  Sampling, JL (Guassian/Sign): Anju
#. FFT-based: Haim
#. Hashing: Vikas




