all: hilbert_train convert2hdf5

hilbert_train: train.cpp BlockADMM.hpp FunctionProx.hpp hilbert.hpp options.hpp io.hpp run.hpp
	g++ -std=c++11 -DSKYLARK_HAVE_OPENMP -DSKYLARK_HAVE_FFTW -DSKYLARK_HAVE_HDF5 -DSKYLARK_HAVE_PROFILER -fopenmp -o hilbert_train train.cpp -lhdf5 -lhdf5_cpp -lelemental -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -lfftw3 -I../
 
convert2hdf5: convert2hdf5.cpp
	g++ -std=c++11 -DSKYLARK_HAVE_HDF5  -o convert2hdf5 convert2hdf5.cpp -lhdf5 -lhdf5_cpp -lelemental -lboost_mpi -lmpich -L${FFTW_ROOT}/lib -lfftw3 -I../

#predict: predict.cpp
#	g++ -o predict predict.cpp -lelemental  -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -I../

clean:
	rm *.o hilbert_train predict convert2hdf5
