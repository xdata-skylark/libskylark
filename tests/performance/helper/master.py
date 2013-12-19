# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI

import sys

if len(sys.argv) != 6:
    print "master.py test_dir total_nr_procs nr_samples data_dir web_dir"
    sys.exit(-1)

if MPI.COMM_WORLD.Get_size() != 1:
    print "Please run the master only on one core. It will spawn slaves automatically."
    sys.exit(-1)


import os
import glob
import commands
from   random import sample

_NUM_SAMPLES = int(sys.argv[3])

try:
    #FIXME: fix nps for all tests?
    nps = sorted(sample(xrange(int(sys.argv[2]) - 1), _NUM_SAMPLES))
except ValueError:
    print "Error: total number of procs must be larger (or equal) to num samples"
else:
    os.chdir(sys.argv[1])
    for infile in glob.glob("*_perf_test.py"):
        for np in nps:
            print "spawning " + str(np + 1)
            #cmd = "mpirun -np %s python %s" % (np + 1, infile)
            #print commands.getoutput(cmd)
            comm = MPI.COMM_SELF.Spawn(sys.executable, args=[infile],
                                       maxprocs=np + 1)

            comm.Barrier()
            comm.Disconnect()

    from generate_plot import generate_plots
    commands.getoutput("mv *.perf %s" % sys.argv[4])
    generate_plots(sys.argv[4], sys.argv[5])

