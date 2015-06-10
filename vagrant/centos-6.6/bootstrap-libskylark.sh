#!/usr/bin/env bash
export NPROC=`nproc`

# Suppressing "dpkg-preconfigure: unable to re-open stdin"
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8
dpkg-reconfigure locales

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

export SKYLARK_SRC_DIR=/home/vagrant/libskylark
export SKYLARK_BUILD_DIR=/home/vagrant/build
export SKYLARK_INSTALL_DIR=/home/vagrant/install

# populate .bashrc
echo "export SKYLARK_SRC_DIR=${SKYLARK_SRC_DIR}" >> ./.bashrc
echo "export SKYLARK_BUILD_DIR=${SKYLARK_BUILD_DIR}" >> ./.bashrc
echo "export SKYLARK_INSTALL_DIR=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "export PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "export PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}" >> ./.bashrc
echo "export LD_LIBRARY_PATH=${SKYLARK_INSTALL_DIR}/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}" >> ./.bashrc
echo "export PATH=${SKYLARK_INSTALL_DIR}/bin:/opt/rh/devtoolset-2/root/usr/bin/:${PATH}" >> ./.bashrc
echo "module load mpich-x86_64" >> ./.bashrc

chown -R vagrant /home/vagrant/.bashrc

# populate .emacs with skylark coding style, in case user wants to use emacs
echo "(load-file \"/home/vagrant/libskylark/doc/script/emacsrc\")"  >> .emacs
chown -R vagrant /home/vagrant/.emacs

# get latest gcc/g++
cd /etc/yum.repos.d
wget http://people.centos.org/tru/devtools-2/devtools-2.repo
yum -y install devtoolset-2-gcc
yum -y install devtoolset-2-binutils
yum -y install devtoolset-2-gcc-gfortran
yum -y install devtoolset-2-gcc-c++
export PATH="/opt/rh/devtoolset-2/root/usr/bin/:$PATH"

# Message Passing Interface
yum -y install mpich mpich-devel mpich-autoload
#XXX: we cannot load the module here, done for the user
export PATH=/usr/lib64/mpich/bin:$PATH

# source control
yum -y install git

# configuration
yum -y install cmake

#OpenBLAS
yum -y install openblas openblas-devel

# BLAS and LAPACK
yum -y install blas blas-devel lapack lapack-devel

# HDF5
yum -y install hdf5 hdf5-devel

#FFTW
yum -y install fftw3 fftw3-devel

# install tools for building documentation
yum -y install doxygen graphviz python-sphinx dvipng

yum -y install unzip



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

    if [ ! -f boost_1_55_0.tar.gz ]; then
        wget http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.gz &> /dev/null
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
    if [ "$calc_md5" == "65ea30b7d2a5be041086ed50bcedc71b" ]; then
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

    boostOk=false
    calc_md5=$(md5sum boost_1_55_0.tar.gz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "93780777cfbf999a600f62883bd54b17" ]; then
        boostOk=true
    else
        rm boost_1_55_0.tar.gz
    fi

    if $randOk && $eleOk && $cbOk && $kdtOk && $spiralOk && $boostOk; then
        break
    fi
    break
done


# Boost
cd $HOME/deps
if [ ! -d "boost_1_55_0" ]; then
    tar xvzf boost_1_55_0.tar.gz &> /dev/null
    cd boost_1_55_0
    ./bootstrap.sh --with-libraries=mpi,serialization,program_options,filesystem,system
    echo "using mpi ;" >> project-config.jam
    ./b2 link=static,shared
    sudo ./b2 install
fi

# Elemental
#FIXME: go to newer hash due to bugs in Elemental!
cd $HOME/deps
if [ ! -d "Elemental-0.86-rc1" ]; then
    unzip 0.86-rc1.zip &> /dev/null
    cd Elemental-0.86-rc1
    rmdir external/metis
    git clone https://github.com/poulson/metis.git external/metis
    mkdir build
    cd build
    cmake -DEL_USE_64BIT_INTS=ON -DCMAKE_BUILD_TYPE=PureRelease -DMATH_LIBS="-L/usr/lib -llapack -lopenblas -lm" ../
    make -j $NPROC
    make install
fi

# CombBLAS
cd $HOME/deps
if [ ! -d "CombBLAS" ]; then
    tar xvfz CombBLAS_beta_14_0.tgz &> /dev/null
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
    tar xvfz kdt-0.3.tar.gz &> /dev/null
    cd kdt-0.3
    yum -y install subversion
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
    tar xvfz Random123-1.08.tar.gz &> /dev/null
    cp -r Random123-1.08/include/Random123 /usr/local/include
fi

# spiral-wht
cd $HOME/deps
if [ ! -d "spiral-wht-1.8" ]; then
    tar xzvf spiral-wht-1.8.tgz &> /dev/null
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
    cd ..
else
    cd libskylark
    git pull
    cd ..
fi

mkdir -p ${SKYLARK_BUILD_DIR}
mkdir -p ${SKYLARK_INSTALL_DIR}
cd ${SKYLARK_BUILD_DIR}
export BLAS_LIBRARIES="-L/usr/lib -lopenblas -lm"
rm CMakeCache.txt
CC=mpicc CXX=mpicxx cmake -DCMAKE_INSTALL_PREFIX=${SKYLARK_INSTALL_DIR} \
                          -DUSE_COMBBLAS=ON ${SKYLARK_SRC_DIR}
make -j $NPROC
make install
make doc

echo "Finished libSkylark Vagrant build.."

# Finalize
cd $HOME
chown -R vagrant /home/vagrant

