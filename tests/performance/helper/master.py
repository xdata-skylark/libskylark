# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI

import os
def create_dir(dirname):
    dir = os.path.dirname(dirname + '/')
    if not os.path.exists(dir):
        os.makedirs(dir)


import argparse
parser = argparse.ArgumentParser(description='Skylark Performance Tests')
parser.add_argument("totalnp" , type=int  , help='total number of processors available')
parser.add_argument("samples" , type=int  , help='number of samples in performance graphs')
parser.add_argument("serve"   , type=bool , help='is this instance serving the tests')
parser.add_argument("testdir" , type=str  , help='path of directory containing tests')
parser.add_argument("webdir"  , type=str  , help='path to directory to output webpages')
parser.add_argument("datadir" , type=str  , help='path of directory containing data')
parser.add_argument("remotes" , type=str  , help='path to file describing remote machines')
args = parser.parse_args()

import sys
if MPI.COMM_WORLD.Get_size() != 1:
    print "Please run the master only on one core. It will spawn slaves automatically."
    sys.exit(-1)


import glob
import commands
from   random import sample

# make sure that the necessary dirs exists and create if not.
create_dir(args.webdir)
create_dir(args.webdir + "/plots")
create_dir(args.datadir)

try:
    #FIXME: fix nps for all tests?
    nps = sorted(sample(xrange(args.totalnp - 1), args.samples))
except ValueError:
    print "Error: total number of procs must be larger (or equal) to num samples"
else:
    os.chdir(args.datadir)

    for infile in glob.glob(args.testdir + "/*_perf_test.py"):
        for np in nps:
            print "spawning " + str(np + 1) + ": " + infile
            #cmd = "mpirun -np %s python %s" % (np + 1, infile)
            #print commands.getoutput(cmd)
            comm = MPI.COMM_SELF.Spawn(sys.executable, args=[infile],
                                       maxprocs=np + 1)

            comm.Barrier()
            comm.Disconnect()

    if args.serve:
        from generate_plot import generate_plots
        generate_plots(args.datadir, args.webdir, args.remotes)

