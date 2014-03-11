all: hilbert_train

hilbert_train: train.cpp BlockADMM.hpp FunctionProx.hpp hilbert.hpp options.hpp io.hpp profiler.hpp
	g++ -std=c++11 -DSKYLARK_HAVE_OPENMP -DSKYLARK_HAVE_FFTW -DSKYLARK_HAVE_HDF5 -DSKYLARK_HAVE_PROFILER -fopenmp -o hilbert_train train.cpp -lhdf5 -lhdf5_cpp -lelemental -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -lfftw3 -I../
 
#convert2hdf5: convert2hdf5.cpp
#	g++ -DSKYLARK_HAVE_HDF5 -o convert2hdf5 convert2hdf5.cpp -lhdf5 -lhdf5_cpp -lelemental -lboost_mpi -lmpich -I../

#predict: predict.cpp
#	g++ -o predict predict.cpp -lelemental  -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -I../

clean:
	rm *.o hilbert_train predict convert2hdf5
