#!/usr/bin/env bash
export NPROC=`nproc`

# Suppressing "dpkg-preconfigure: unable to re-open stdin"
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8
dpkg-reconfigure locales

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# populate .bashrc
echo "export SKYLARK_SRC_DIR=/home/vagrant/libskylark" > ./.bashrc
echo "export SKYLARK_BUILD_DIR=/home/vagrant/build" >> ./.bashrc
echo "export SKYLARK_INSTALL_DIR=/home/vagrant/install" >> ./.bashrc
echo "export PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "export PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}" >> ./.bashrc
echo "export LD_LIBRARY_PATH=${SKYLARK_INSTALL_DIR}/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}" >> ./.bashrc
#echo "export FFTW_ROOT=/usr/lib/x86_64-linux-gnu/" >> ./.bashrc
chown -R vagrant /home/vagrant/.bashrc

# make sure the package information is up-to-date
apt-get update

# python development
apt-get install -y python-dev python-setuptools

# compilers
apt-get install -y g++ gfortran

# Message Passing Interface
apt-get install -y libcr-dev mpich2

# source control
apt-get install -y git

# configuration
apt-get install -y cmake

# BLAS and LAPACK
apt-get install -y libblas-dev
apt-get install -y libblas3gf
apt-get install -y liblapack-dev
apt-get install -y liblapack3gf

#OpenBLAS
apt-get install -y libopenblas-base libopenblas-dev

# Boost (FIXME: we could only install selected packages)
apt-get install -y libboost-all-dev

# Numpy and Scipy stack installation as recommended at scipy.org
apt-get install -y python-numpy python-scipy python-matplotlib ipython \
                   ipython-notebook python-pandas python-sympy python-nose


# mpi4py
easy_install mpi4py

# HDF5
apt-get install -y libhdf5-serial-dev
easy_install h5py

#FFTW
#libfftw3-mpi3
apt-get install -y libfftw3-dev libfftw3-mpi-dev

# install tools for building documentation
apt-get install -y doxygen graphviz python-sphinx dvipng

#FIXME SPHINX extensions?

# numpydoc
easy_install numpydoc

apt-get install -y unzip


# download software dependencies that we have to build..
# make sure we have all the files! before we continue
mkdir -p $HOME/deps
cd $HOME/deps
while true; do
    if [ ! -f Random123-1.08.tar.gz ]; then
        wget http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz &> /dev/null
    fi

    if [ ! -f 0.86-rc1.zip ]; then
        wget https://github.com/elemental/Elemental/archive/0.86-rc1.zip &> /dev/null
    fi

    if [ ! -f CombBLAS_beta_14_0.tgz ]; then
        wget http://gauss.cs.ucsb.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_14_0.tgz &> /dev/null
    fi

    if [ ! -f kdt-0.3.tar.gz ]; then
        wget http://sourceforge.net/projects/kdt/files/kdt-0.3.tar.gz &> /dev/null
    fi

    if [ ! -f spiral-wht-1.8.tgz ]; then
        wget http://www.ece.cmu.edu/~spiral/software/spiral-wht-1.8.tgz &> /dev/null
    fi

    randOk=false
    calc_md5=$(md5sum Random123-1.08.tar.gz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "87d2783831c7a95b244868bf754a7f50" ]; then
        randOk=true
    else
        rm Random123-1.08.tar.gz
    fi

    eleOk=false
    calc_md5=$(md5sum 0.86-rc1.zip | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "6422a203bd3941c962add1543528bb78" ]; then
        eleOk=true
    else
        rm 0.86-rc1.zip
    fi

    cbOk=false
    calc_md5=$(md5sum CombBLAS_beta_14_0.tgz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "57aed213d7e794153ea29465559e7cfa" ]; then
        cbOk=true
    else
        rm CombBLAS_beta_14_0.tgz
    fi

    kdtOk=false
    calc_md5=$(md5sum kdt-0.3.tar.gz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "c15c58a6457397426bc95c0f1892a93d" ]; then
        kdtOk=true
    else
        rm kdt-0.3.tar.gz
    fi

    spiralOk=false
    calc_md5=$(md5sum spiral-wht-1.8.tgz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "28b4d854025b42df4af9616097809726" ]; then
        spiralOk=true
    else
        rm spiral-wht-1.8.tgz
    fi

    if $randOk && $eleOk && $cbOk && $kdtOk && $spiralOk; then
        break
    fi
done


# Elemental
cd $HOME/deps
if [ ! -d "Elemental-0.86-rc1" ]; then
    unzip 0.86-rc1.zip
    cd Elemental-0.86-rc1
    rmdir external/metis
    git clone https://github.com/poulson/metis.git external/metis
    mkdir build
    cd build
    cmake -DEL_USE_64BIT_INTS=ON -DCMAKE_BUILD_TYPE=HybridRelease -DMATH_LIBS="-L/usr/lib -llapack -lopenblas -lm" ../
    make -j $NPROC
    make install
fi

# CombBLAS
cd $HOME/deps
if [ ! -d "CombBLAS" ]; then
    tar xvfz CombBLAS_beta_14_0.tgz
    cd CombBLAS/
    cp /vagrant/combblas.patch .
    git apply --ignore-space-change --ignore-whitespace combblas.patch
    rm combblas.patch
    cmake .
    make -j $NPROC
    cp *.so /usr/local/lib
    mkdir /usr/local/include/CombBLAS
    #XXX: ugly but CombBLAS cannot be installed in an other way..
    cp *.h /usr/local/include/CombBLAS
    cp *.cpp /usr/local/include/CombBLAS
    cp -R SequenceHeaps /usr/local/include/CombBLAS
    cp -R psort-1.0 /usr/local/include/CombBLAS
    cp -R graph500-1.2 /usr/local/include/CombBLAS
fi

# KDT
cd $HOME/deps
if [ ! -d "kdt-0.3" ]; then
    tar xvfz kdt-0.3.tar.gz
    cd kdt-0.3
    apt-get install subversion
    export CC=mpicxx
    export CXX=mpicxx
    python ./setup.py build
    python ./setup.py install
    cd ..
    unset CC
    unset CXX
fi

# Random123
cd $HOME/deps
if [ ! -d "Random123-1.08" ]; then
    tar xvfz Random123-1.08.tar.gz
    cp -r Random123-1.08/include/Random123 /usr/local/include
fi

# spiral-wht
cd $HOME/deps
if [ ! -d "spiral-wht-1.8" ]; then
    tar xzvf spiral-wht-1.8.tgz
    cd spiral-wht-1.8/
    ./configure CFLAGS="-fPIC -fopenmp" --enable-RAM=16000 --enable-DDL --enable-IL --enable-PARA=8
    make -j $NPROC
    make install
    #/usr/local/bin/wht_dp.prl
fi


# To build libSkylark (everything enabled):
cd $HOME
source /home/vagrant/.bashrc
cd /home/vagrant

if [ ! -d "libskylark" ]; then
    yes | git clone https://github.com/xdata-skylark/libskylark.git
    cd libskylark
    git checkout development
    cp /vagrant/find_fftw.patch .
    git apply --ignore-space-change --ignore-whitespace find_fftw.patch
    rm find_fftw.patch
    cd ..
else
    cd libskylark
    git pull
    cd ..
fi

mkdir -p ${SKYLARK_BUILD_DIR}
mkdir -p ${SKYLARK_INSTALL_DIR}
cd ${SKYLARK_BUILD_DIR}
rm CMakeCache.txt
CC=mpicc CXX=mpicxx cmake -DCMAKE_INSTALL_PREFIX=${SKYLARK_INSTALL_DIR} \
                          -DUSE_COMBBLAS=ON ${SKYLARK_SRC_DIR}
make -j $NPROC
make install
make doc

# Finalize
chown -R vagrant /home/vagrant

echo "Finished libSkylark Vagrant build.."

apt-get install dtach
ipython profile create nbserver

echo "c = get_config()" > $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.IPKernelApp.pylab = 'inline'  # if you want plotting support always" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.NotebookApp.ip = '*'" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py

cd /home/vagrant
dtach -n /tmp/ipython_notebook -Ez ipython notebook --profile=nbserver

#FIXME: dir to /home/vagrant, run as vagrant user?
echo "dtach -n /tmp/ipython_notebook -Ez ipython notebook --profile=nbserver" > /etc/rc.local
echo "exit 0" >> /etc/rc.local

echo "Started IPython notebook. Point your browser to http://127.0.0.1:6868"
