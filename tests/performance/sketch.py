from mpi4py import MPI
import elem, skylark, time

# Parameters
m  = 200000
n  = 5000
sm = 20000
sn = 2000
seed = 123836

elem.Initialize()
ctxt = skylark.Context(123836)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (rank == 0):
    print "************ Basic skylark/sketch performance tests (script date: April-23, 2013)."

# VR_STAR, tall-and-skinny
grid = elem.Grid()
A = elem.DistMat_VR_STAR( grid )
A.Resize(m, n)
localHeight = A.LocalHeight()
localWidth = A.LocalWidth()
colShift = A.ColShift()
rowShift = A.RowShift()
colStride = A.ColStride()
rowStride = A.RowStride()
data = A.Data()
ldim = A.LDim()
for jLocal in xrange(0,localWidth):
  j = rowShift + jLocal*rowStride
  for iLocal in xrange(0,localHeight):
    i = colShift + iLocal*colStride
    data[iLocal+jLocal*ldim] = i-j

# VC_STAR, tall-and-skinny
grid = elem.Grid()
B = elem.DistMat_VC_STAR( grid )
B.Resize(m, n)
localHeight = B.LocalHeight()
localWidth = B.LocalWidth()
colShift = B.ColShift()
rowShift = B.RowShift()
colStride = B.ColStride()
rowStride = B.RowStride()
data = B.Data()
ldim = B.LDim()
for jLocal in xrange(0,localWidth):
  j = rowShift + jLocal*rowStride
  for iLocal in xrange(0,localHeight):
    i = colShift + iLocal*colStride
    data[iLocal+jLocal*ldim] = i-j

if rank == 0:
    print "********* Sketching tall-and-skinny matrices:"

if rank == 0:
    print "****** Columnwise sketching:"

# JLT, VR_STAR, columnwise
t = MPI.Wtime()
S = skylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", m, sm)
SA = elem.Mat()
SA.Resize(sm, n)
S.Apply(A, SA, 1)
S.Free()
SA.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> JLT on tall-and-skinny VR_STAR, columnwise - ", telp

# JLT, VC_STAR, columnwise
t = MPI.Wtime()
S = skylark.JLT(ctxt, "DistMatrix_VC_STAR", "Matrix", m, sm)
SB = elem.Mat()
SB.Resize(sm, n)
S.Apply(B, SB, 1)
S.Free()
SB.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> JLT on tall-and-skinny VC_STAR, columnwise - ", telp

# FJLT, VR_STAR, columnwise
t = MPI.Wtime()
S = skylark.FJLT(ctxt, "DistMatrix_VR_STAR", "Matrix", m, sm)
SA = elem.Mat()
SA.Resize(sm, n)
S.Apply(A, SA, 1)
S.Free()
SA.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> FJLT on tall-and-skinny VR_STAR, columnwise - ", telp

# FJLT, VC_STAR, columnwise
t = MPI.Wtime()
S = skylark.FJLT(ctxt, "DistMatrix_VC_STAR", "Matrix", m, sm)
SB = elem.Mat()
SB.Resize(sm, n)
S.Apply(B, SB, 1)
S.Free()
SB.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> FJLT on tall-and-skinny VC_STAR, columnwise - ", telp


if rank == 0:
    print "****** Rowwise sketching:"

# JLT, VR_STAR, rowwise
t = MPI.Wtime()
S = skylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", n, sn)
SA = elem.Mat()
SA.Resize(m, sn)
S.Apply(A, SA, 2)
S.Free()
SA.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> JLT on tall-and-skinny VR_STAR, rowwise - ", telp

# JLT, VC_STAR, rowwise
t = MPI.Wtime()
S = skylark.JLT(ctxt, "DistMatrix_VC_STAR", "Matrix", n, sn)
SB = elem.Mat()
SB.Resize(m, sn)
S.Apply(B, SB, 2)
S.Free()
SB.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> JLT on tall-and-skinny VC_STAR, rowwise - ", telp

# FJLT, VR_STAR, rowwise
t = MPI.Wtime()
S = skylark.FJLT(ctxt, "DistMatrix_VR_STAR", "Matrix", n, sn)
SA = elem.Mat()
SA.Resize(m, sn)
S.Apply(A, SA, 2)
S.Free()
SA.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> FJLT on tall-and-skinny VR_STAR, rowwise - ", telp

# FJLT, VC_STAR, rowwise
t = MPI.Wtime()
S = skylark.FJLT(ctxt, "DistMatrix_VC_STAR", "Matrix", n, sn)
SB = elem.Mat()
SB.Resize(m, sn)
S.Apply(B, SB, 2)
S.Free()
SB.Free()
telp = MPI.Wtime() - t
if rank == 0:
    print ">>> FJLT on tall-and-skinny VC_STAR, rowwise - ", telp
