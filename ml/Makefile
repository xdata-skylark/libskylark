all: hilbert_train predict

hilbert_train: train.cpp BlockADMM.hpp FunctionProx.hpp hilbert.hpp options.hpp io.hpp
	g++ -fopenmp -o hilbert_train train.cpp -lelemental -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -lfftw3 -I../

predict: predict.cpp
	g++ -o predict predict.cpp -lelemental  -lboost_mpi -lboost_program_options -L${FFTW_ROOT}/lib -lmpich -I../

clean:
	rm *.o hilbert_train predict
