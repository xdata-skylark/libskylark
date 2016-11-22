## This is executed on a fresh Ubuntu Trusty.
## Assumes running with sudo.

#yes | apt-get update

yes | apt-get install git

locale-gen UTF-8
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8
dpkg-reconfigure locales

yes | apt-get install cmake libcr-dev cython
yes | apt-get install python-setuptools python-matplotlib
yes | apt-get install ipython ipython-notebook
yes | apt-get install python-pandas python-sympy python-nose
yes | apt-get install swig swig-examples swig2.0-examples
yes | apt-get install doxygen graphviz python-sphinx dvipng unzip subversion maven
yes | apt-get install libz-dev
yes | apt-get install mpich

# Install HDF5 (TODO: maybe parallel too?)
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz
tar zxvf hdf5-1.8.17.tar.gz
cd hdf5-1.8.17/
./configure --prefix=/usr/local --enable-cxx
make -j2
make install
cd ..
rm -rf hdf5-*

# easy-install some pacakges
easy_install mpi4py
easy_install h5py
easy_install networkx

# Install Random123
wget http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz
tar zxvf Random123-1.08.tar.gz
cp -r Random123-1.08/include/Random123 /usr/local/include
rm -rf Random123-1.08*

# Install Boost
wget http://downloads.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.gz
tar zxvf boost_1_60_0.tar.gz
cd boost_1_60_0/
./bootstrap.sh --with-libraries=mpi,python,random,serialization,program_options,system,filesystem
echo "using mpi ;" >> project-config.jam
./b2 -j 2 link=static,shared
./b2 install
cd ..
rm -rf boost*

# Install FFTW
wget http://www.fftw.org/fftw-3.3.4.tar.gz
tar zxvf fftw-3.3.4.tar.gz
cd fftw-3.3.4/
./configure --enable-single --enable-mpi --enable-shared
make -j 2
make install
./configure --enable-mpi --enable-shared
make -j 2
make install
cd ..
rm -rf fftw*

# Install SPIRAL
wget http://www.ece.cmu.edu/~spiral/software/spiral-wht-1.8.tgz
tar zxvf spiral-wht-1.8.tgz
cd spiral-wht-1.8
./configure CC=gcc CFLAGS="-fPIC -fopenmp" PCFLAGS="-fPIC -fopenmp" --enable-RAM=16000 --enable-DDL --enable-IL --enable-PARA=16
make -j2
make install
cd ..
rm -rf spiral*

# Install OpenBLAS
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.15.tar.gz
tar zxvf v0.2.15.tar.gz
cd OpenBLAS-0.2.15/
make USE_OPENMP=1 FC=gfortran
make PREFIX=/usr/local install
cd ..
rm -rf OpenBLAS-0.2.15/
rm v0.2.15.tar.gz

# Install LAPACK
wget http://www.netlib.org/lapack/lapack-3.6.0.tgz
tar zxvf lapack-3.6.0.tgz
cd lapack-3.6.0/
mkdir build; cd build
cmake -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON -DBLAS_LIBRARIES="-L/usr/local/lib -lopenblas -lm" ..
make -j2
make install
cd ../..
rm -rf lapack*

# Install Elemental
git clone https://github.com/elemental/Elemental.git
cd Elemental
git checkout tags/v0.87.2
mkdir build; cd build
cmake -DCMAKE_INSTALL_LIBDIR=lib -DEL_USE_64BIT_INTS=ON -DEL_HAVE_QUADMATH=OFF -DCMAKE_BUILD_TYPE=Release -DEL_HYBRID=ON -DBUILD_SHARED_LIBS=ON -DINSTALL_PYTHON_PACKAGE=ON -DMATH_LIBS="-L/usr/local/lib -llapack -lopenblas -lm" ../
make -j2
make install

# Elemental is not deleted to allow easy updates.

# Clone skylark and compile
git clone https://github.com/xdata-skylark/libskylark.git
cd libskylark
git checkout development
mkdir build
cd build
export LAPACK_LIBRARIES="-L/usr/local/lib -lopenblas -lm"
export BLAS_LIBRARIES="-L/usr/local/lib -lopenblas -lm"
CC=mpicc CXX=mpicxx cmake -DBUILD_EXAMPLES=ON -DUSE_FFTW=ON ..
make -j2
