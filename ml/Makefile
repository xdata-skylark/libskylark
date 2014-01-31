hilbert_train: train.cpp BlockADMM.hpp FeatureTransform.hpp FunctionProx.hpp matrixutils.hpp hilbert.hpp options.hpp
	g++ -o hilbert_train train.cpp -lelemental -lboost_mpi -lboost_program_options -L/home/vikas/research/software/fftw-3.3.3/install/lib -lmpich -lfftw3
clean:
	rm *.o hilbert_train
