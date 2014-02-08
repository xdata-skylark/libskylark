all: hilbert_train predict

hilbert_train: train.cpp BlockADMM.hpp FeatureTransform.hpp FunctionProx.hpp hilbert.hpp options.hpp io.hpp
	g++ -o hilbert_train train.cpp -lelemental -lboost_mpi -lboost_program_options -L/home/vikas/research/software/fftw-3.3.3/install/lib -lmpich -lfftw3

predict: predict.cpp
	g++ -o predict predict.cpp -lelemental  -lboost_mpi -lboost_program_options -L/home/vikas/research/software/fftw-3.3.3/install/lib -lmpich

clean:
	rm *.o hilbert_train predict
