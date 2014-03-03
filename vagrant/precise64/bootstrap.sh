#!/usr/bin/env bash

# export to sudo environment
# XXX need to clean environment variables export
export SKYLARK_INSTALL_DIR=/home/vagrant/install
export COMBBLAS_ROOT=/home/vagrant/CombBLAS_beta_13_0
export PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}
export PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# populate .bashrc
# XXX need to clean environment variables export
echo "SKYLARK_SRC_DIR=/home/vagrant/skylark/tools/analytics/ibm/skylark" > ./.bashrc
echo "SKYLARK_BUILD_DIR=/home/vagrant/build" >> ./.bashrc
echo "SKYLARK_INSTALL_DIR=/home/vagrant/install" >> ./.bashrc
echo "COMBBLAS_ROOT=/home/vagrant/CombBLAS_beta_13_0" >> ./.bashrc
echo "PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}" >> ./.bashrc
echo "LD_LIBRARY_PATH=${SKYLARK_INSTALL_DIR}/lib:${COMBBLAS_ROOT}/lib:/usr/local/lib" >> ./.bashrc
echo "export SKYLARK_SRC_DIR" >> ./.bashrc
echo "export SKYLARK_BUILD_DIR" >> ./.bashrc
echo "export SKYLARK_INSTALL_DIR" >> ./.bashrc
echo "export COMBBLAS_ROOT" >> ./.bashrc
echo "export PYTHON_SITE_PACKAGES" >> ./.bashrc
echo "export PYTHONPATH" >> ./.bashrc
echo "export LD_LIBRARY_PATH" >> ./.bashrc
chown -R vagrant /home/vagrant/.bashrc

# python development
apt-get install -y python-dev

# make sure the package information is up-to-date
apt-get update

# setuptools
apt-get install -y python-setuptools

# compilers
apt-get install -y g++ 
apt-get install -y gfortran

# source control
apt-get install -y git

# configuration
wget http://www.cmake.org/files/v2.8/cmake-2.8.11.2.tar.gz
tar xvfz cmake-2.8.11.2.tar.gz
cd cmake-2.8.11.2/
mkdir build
cd build
../bootstrap
make
make install
cd ../..

# BLAS and LAPACK
apt-get install -y libblas-dev 
apt-get install -y libblas-doc 
apt-get install -y libblas3gf 
apt-get install -y liblapack-dev 
apt-get install -y liblapack-doc 
apt-get install -y liblapack3gf
 
# OpenBLAS
# XXX as alternative to plain vanilla BLAS; now also built
wget -O OpenBLAS.tgz http://github.com/xianyi/OpenBLAS/tarball/v0.2.8
tar xzvf OpenBLAS.tgz
cd xianyi-OpenBLAS-9c51cdf/
make -j4
make PREFIX=/usr/local install
cd ..

# Message Passing Interface
apt-get install -y libcr-dev 
apt-get install -y mpich2 
apt-get install -y mpich2-doc

# numpy
wget http://downloads.sourceforge.net/project/numpy/NumPy/1.7.0/numpy-1.7.0.tar.gz
tar xvzf numpy-1.7.0.tar.gz
cd numpy-1.7.0
echo "[atlas]" > site.cfg
echo "atlas_libs = openblas" >> site.cfg
echo "library_dirs = /usr/local/lib" >> site.cfg
./setup.py build
./setup.py install
cd ..

# scipy
wget http://downloads.sourceforge.net/project/scipy/scipy/0.12.0/scipy-0.12.0.tar.gz
tar xvzf scipy-0.12.0.tar.gz
cd scipy-0.12.0/
cp ../numpy-1.7.0/site.cfg .
./setup.py build
./setup.py install
cd ..

# SWIG
apt-get install -y swig 
apt-get install -y swig-doc 
apt-get install -y swig-examples 
apt-get install -y swig2.0-examples 
apt-get install -y swig2.0-doc

# mpi4py
easy_install mpi4py

# HDF5
apt-get install -y libhdf5-serial-dev
easy_install h5py

# Boost
wget http://sourceforge.net/projects/boost/files/boost/1.53.0/boost_1_53_0.tar.gz
tar xvfz boost_1_53_0.tar.gz
cd boost_1_53_0
./bootstrap.sh --with-libraries=mpi,python,random,serialization
echo "using mpi ;" >> project-config.jam
./b2 link=static,shared
./b2 install
cd ..

# Elemental
wget http://libelemental.org/pub/releases/elemental-0.81.tgz
tar xvfz elemental-0.81.tgz
cd elemental-0.81/
mkdir build
cd build
cmake -D USE_SWIG=ON ..
make -j4
make install
cp *.py /usr/local/lib/python2.7/dist-packages/
cp _*.so /usr/local/lib/python2.7/dist-packages/
cd ../..

# CombBLAS
wget http://gauss.cs.ucsb.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_13_0.tgz
tar xvfz CombBLAS_beta_13_0.tgz
cd CombBLAS_beta_13_0/
cp /vagrant/combblas.patch .
git apply --ignore-space-change --ignore-whitespace combblas.patch
rm combblas.patch
cmake .
make
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
make -j4
make install
cd ..

# Random123
wget http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz
tar xvfz Random123-1.08.tar.gz
cp -r Random123-1.08/include/Random123 /usr/local/include

# spiral-wht
wget http://www.ece.cmu.edu/~spiral/software/spiral-wht-1.8.tgz
tar xzvf spiral-wht-1.8.tgz
cd spiral-wht-1.8/
./configure CFLAGS="-fPIC -fopenmp" --enable-RAM=16000 --enable-DDL --enable-IL --enable-PARA=8
make -j4
make install
#/usr/local/bin/wht_dp.prl
cd ..

# doxygen
apt-get install -y doxygen

# graphviz
apt-get install -y graphviz

# sphinx
apt-get install -y python-sphinx 

# matplotlib
apt-get install -y python-matplotlib

# numpydoc
easy_install numpydoc

# dvipng
apt-get install -y dvipng


# To build skylark (with CombBLAS support):
#
# copy public/private keys pair under ~/.ssh
#
# git clone git@azmodan.watson.ibm.com:skylark.git
# chown -R vagrant ../skylark
# cd skylark
# git checkout skylark-development
#
# mkdir ${SKYLARK_BUILD_DIR}
# mkdir ${SKYLARK_INSTALL_DIR}
# cd ${SKYLARK_BUILD_DIR}
# CXX=mpicc cmake -DCMAKE_INSTALL_PREFIX=${SKYLARK_INSTALL_DIR} -DUSE_COMBBLAS=ON ${SKYLARK_SRC_DIR}
# make
# make install



