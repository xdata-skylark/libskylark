# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI

import sys

if len(sys.argv) != 5:
    print "master.py input_dir total_nr_procs nr_samples output_dir"
    sys.exit(-1)

if MPI.COMM_WORLD.Get_size() != 1:
    print "Please run the master only on one core. It will spawn slaves automatically."
    sys.exit(-1)


import os
import glob
from   random import sample

_NUM_SAMPLES = int(sys.argv[3])

try:
    #FIXME: fix nps for all tests?
    nps = sample(xrange(int(sys.argv[2]) - 1), _NUM_SAMPLES)
except ValueError:
    print "Error: total number of procs must be larger (or equal) to num samples"
else:
    os.chdir(sys.argv[1])
    for infile in glob.glob("*_perf_test.py"):
        for np in nps:
            print "spawning " + str(np + 1)
            comm = MPI.COMM_SELF.Spawn(sys.executable, args=[infile],
                                       maxprocs=np + 1)

            comm.Barrier()
            comm.Disconnect()

    from generate_plot import generate_plots
    generate_plots(sys.argv[1], sys.argv[4])

