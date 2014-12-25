#!/usr/bin/env bash

# export to sudo environment
# XXX need to clean environment variables export
export SKYLARK_INSTALL_DIR=/home/vagrant/install
export COMBBLAS_ROOT=/home/vagrant/CombBLAS
export PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}
export PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
export NPROC=`nproc`

# populate .bashrc
# XXX need to clean environment variables export
echo "SKYLARK_SRC_DIR=/home/vagrant/libskylark" > ./.bashrc
echo "SKYLARK_BUILD_DIR=/home/vagrant/build" >> ./.bashrc
echo "SKYLARK_INSTALL_DIR=/home/vagrant/install" >> ./.bashrc
echo "COMBBLAS_ROOT=/home/vagrant/CombBLAS" >> ./.bashrc
echo "PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}" >> ./.bashrc
echo "LD_LIBRARY_PATH=${SKYLARK_INSTALL_DIR}/lib:${COMBBLAS_ROOT}:/usr/local/lib" >> ./.bashrc
echo "export SKYLARK_SRC_DIR" >> ./.bashrc
echo "export SKYLARK_BUILD_DIR" >> ./.bashrc
echo "export SKYLARK_INSTALL_DIR" >> ./.bashrc
echo "export COMBBLAS_ROOT" >> ./.bashrc
echo "export PYTHON_SITE_PACKAGES" >> ./.bashrc
echo "export PYTHONPATH" >> ./.bashrc
echo "export LD_LIBRARY_PATH" >> ./.bashrc
echo "export LD_PRELOAD=/home/vagrant/CombBLAS/libMPITypelib.so:/home/vagrant/CombBLAS/libCommGridlib.so" >> ./.bashrc
chown -R vagrant /home/vagrant/.bashrc

# make sure the package information is up-to-date
apt-get update

# python development
apt-get install -y python-dev

# setuptools
apt-get install -y python-setuptools

# compilers
apt-get install -y g++
apt-get install -y gfortran

# source control
apt-get install -y git

# configuration
apt-get install -y cmake

# BLAS and LAPACK
apt-get install -y libblas-dev
apt-get install -y libblas3gf
apt-get install -y liblapack-dev
apt-get install -y liblapack3gf

# OpenBLAS
# XXX as alternative to plain vanilla BLAS; now also built
wget -O OpenBLAS.tgz http://github.com/xianyi/OpenBLAS/tarball/v0.2.8
tar xzvf OpenBLAS.tgz
cd xianyi-OpenBLAS-9c51cdf/
make -j $NPROC
make PREFIX=/usr/local install
cd ..

# Message Passing Interface
apt-get install -y libcr-dev
apt-get install -y mpich2

# Numpy and Scipy stack installation as recommended at scipy.org
apt-get install -y python-numpy 
apt-get install -y python-scipy 
apt-get install -y python-matplotlib 
apt-get install -y ipython 
apt-get install -y ipython-notebook 
apt-get install -y python-pandas 
apt-get install -y python-sympy 
apt-get install -y python-nose


# SWIG
apt-get install -y swig
apt-get install -y swig-examples
apt-get install -y swig2.0-examples

# mpi4py
easy_install mpi4py

# HDF5
apt-get install -y libhdf5-serial-dev
easy_install h5py

# Boost
wget http://sourceforge.net/projects/boost/files/boost/1.53.0/boost_1_53_0.tar.gz
tar xvfz boost_1_53_0.tar.gz
cd boost_1_53_0
./bootstrap.sh --with-libraries=mpi,python,random,serialization,program_options
echo "using mpi ;" >> project-config.jam
./b2 link=static,shared
./b2 install
cd ..

# Elemental
wget http://libelemental.org/pub/releases/Elemental-0.85.tgz
tar xvfz Elemental-0.85.tgz
cd Elemental-0.85/
mkdir build
cd build
cmake -D MATH_LIBS="-L/usr/local/lib -llapack -lopenblas -lm" ....
make -j $NPROC
make install
cd ../..

# CombBLAS
wget http://gauss.cs.ucsb.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_14_0.tgz
tar xvfz CombBLAS_beta_14_0.tgz
cd CombBLAS/
cp /vagrant/combblas.patch .
git apply --ignore-space-change --ignore-whitespace combblas.patch
rm combblas.patch
cmake .
make -j $NPROC
cp *.so /usr/local/lib
mkdir /usr/local/include/CombBLAS
cp *.h /usr/local/include/CombBLAS
cd ..

# KDT
wget http://sourceforge.net/projects/kdt/files/kdt-0.3.tar.gz
tar xvfz kdt-0.3.tar.gz
cd kdt-0.3
export CC=mpicxx
export CXX=mpicxx
python ./setup.py build
python ./setup.py install
cd ..

# FFTW
wget http://www.fftw.org/fftw-3.3.3.tar.gz
tar xvfz fftw-3.3.3.tar.gz
cd fftw-3.3.3/
./configure --enable-shared
make -j $NPROC
make install
cd ..

# Random123
wget http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz
tar xvfz Random123-1.08.tar.gz
cp -r Random123-1.08/include/Random123 /usr/local/include

# doxygen
apt-get install -y doxygen

# graphviz
apt-get install -y graphviz

# sphinx
apt-get install -y python-sphinx

# numpydoc
easy_install numpydoc

# dvipng
apt-get install -y dvipng


