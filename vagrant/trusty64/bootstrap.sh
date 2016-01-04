#!/usr/bin/env bash
export NPROC=`nproc`
export UNAME=vagrant

# Suppressing "dpkg-preconfigure: unable to re-open stdin"
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8
dpkg-reconfigure locales

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

export SKYLARK_SRC_DIR=/home/${UNAME}/libskylark
export SKYLARK_BUILD_DIR=/home/${UNAME}/build
export SKYLARK_INSTALL_DIR=/home/${UNAME}/install
export LIBHDFS_ROOT=/home/${UNAME}/deps/hadoop-2.7.0
export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

# populate .bashrc
echo "export SKYLARK_SRC_DIR=${SKYLARK_SRC_DIR}" >> ./.bashrc
echo "export SKYLARK_BUILD_DIR=${SKYLARK_BUILD_DIR}" >> ./.bashrc
echo "export SKYLARK_INSTALL_DIR=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "export LIBHDFS_ROOT=/home/${UNAME}/deps/hadoop-2.7.0" >> ./.bashrc
echo "export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64" >> ./.bashrc
echo "export PYTHON_SITE_PACKAGES=${SKYLARK_INSTALL_DIR}" >> ./.bashrc
echo "export PYTHONPATH=${SKYLARK_INSTALL_DIR}/lib/python2.7/site-packages:${PYTHONPATH}" >> ./.bashrc
echo "export LD_LIBRARY_PATH=${SKYLARK_INSTALL_DIR}/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}" >> ./.bashrc
echo "export PATH=${SKYLARK_INSTALL_DIR}/bin:${PATH}" >> ./.bashrc
echo "export LD_PRELOAD=/usr/lib/libmpi.so" >> ./.bashrc
#echo "export FFTW_ROOT=/usr/lib/x86_64-linux-gnu/" >> ./.bashrc

chown -R ${UNAME} /home/${UNAME}/.bashrc

# populate .emacs with skylark coding style, in case user wants to use emacs
echo "(load-file \"/home/${UNAME}/libskylark/doc/script/emacsrc\")"  >> .emacs

chown -R ${UNAME} /home/${UNAME}/.emacs

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

# Java for HADOOP
apt-get install -y openjdk-7-jdk

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

    if [ ! -f CombBLAS_beta_14_0.tgz ]; then
        wget http://gauss.cs.ucsb.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_14_0.tgz &> /dev/null
    fi

    if [ ! -f kdt-0.3.tar.gz ]; then
        wget http://sourceforge.net/projects/kdt/files/kdt-0.3.tar.gz &> /dev/null
    fi

    if [ ! -f spiral-wht-1.8.tgz ]; then
        wget http://www.ece.cmu.edu/~spiral/software/spiral-wht-1.8.tgz &> /dev/null
    fi

    if [ ! -f hadoop-2.7.0.tar.gz ]; then
        wget http://mirror.sdunix.com/apache/hadoop/common/hadoop-2.7.0/hadoop-2.7.0.tar.gz &> /dev/null
    fi

    randOk=false
    calc_md5=$(md5sum Random123-1.08.tar.gz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "87d2783831c7a95b244868bf754a7f50" ]; then
        randOk=true
    else
        rm Random123-1.08.tar.gz
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

    hadoopOk=false
    calc_md5=$(md5sum hadoop-2.7.0.tar.gz | /usr/bin/cut -f 1 -d " ")
    if [ "$calc_md5" == "79a6e87b09011861309c153a856c3ca1" ]; then
        hadoopOk=true
    else
        rm hadoop-2.7.0.tar.gz
    fi

    if $randOk && $cbOk && $kdtOk && $spiralOk & $hadoopOk ; then
        break
    fi
done


# Elemental
cd $HOME/deps
if [ ! -d "Elemental" ]; then
    git clone https://github.com/elemental/Elemental.git
    cd Elemental
    git checkout a27f9f0a6fc33a971dd81c61f6fbe3ce71f61814
    mkdir build
    cd build
    cmake -DEL_USE_64BIT_INTS=ON -DCMAKE_BUILD_TYPE=Release -DEL_HYBRID=ON -DBUILD_SHARED_LIBS=ON -DMATH_LIBS="-L/usr/lib -llapack -lopenblas -lm" ../
    make -j $NPROC 1> /dev/null
    make install 1> /dev/null
fi

# CombBLAS
cd $HOME/deps
if [ ! -d "CombBLAS" ]; then
    tar xvfz CombBLAS_beta_14_0.tgz &> /dev/null
    cd CombBLAS/
    echo """
diff --git a/RefGen21.h b/RefGen21.h
index b8c7974..f93592c 100644
--- a/RefGen21.h
+++ b/RefGen21.h
@@ -134,7 +134,7 @@ public:

        /* 32-bit code */
        uint32_t h = (uint32_t)(x >> 32);
-       uint32_t l = (uint32_t)(x & UINT32_MAX);
+       uint32_t l = (uint32_t)(x & std::numeric_limits<uint32_t>::max());
        #ifdef USE_GCC_BYTESWAP
         h = __builtin_bswap32(h);
         l = __builtin_bswap32(l);
""" | git apply --ignore-space-change --ignore-whitespace
    cmake -DBUILD_SHARED_LIBS=ON .
    make -j $NPROC 1> /dev/null
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
    apt-get install subversion
    export CC=mpicxx
    export CXX=mpicxx
    python ./setup.py build 1> /dev/null
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
    make -j $NPROC 1> /dev/null
    make install 1> /dev/null
    #/usr/local/bin/wht_dp.prl
fi

# hadoop
cd $HOME/deps
if [ ! -d "hadoop-2.7.0" ]; then
    mkdir /home/${UNAME}/deps
    tar xzvf hadoop-2.7.0.tar.gz -C /home/${UNAME}/deps 1> /dev/null
fi


# To build libSkylark (everything enabled):
cd $HOME
source /home/${UNAME}/.bashrc
cd /home/${UNAME}

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
make doc 1> /dev/null

echo "Finished libSkylark ${UNAME} build.."

# Prepare notebook directory
mkdir /home/${UNAME}/notebooks
mkdir /home/${UNAME}/notebooks/data
cd mkdir /home/${UNAME}/notebooks/data
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2 &> /dev/null
bzip2 -d usps.bz2 &> /dev/null
cp ${SKYLARK_SRC_DIR}/python-skylark/skylark/notebooks/* /home/${UNAME}/notebooks/

# Finalize
cd $HOME
chown -R ${UNAME} /home/${UNAME}


apt-get install -y dtach
ipython profile create nbserver

echo "c = get_config()" > $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.IPKernelApp.pylab = 'inline'" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.NotebookApp.ip = '*'" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py
echo "c.NotebookManager.notebook_dir = u'/home/${UNAME}/notebooks'" >> $HOME/.ipython/profile_nbserver/ipython_notebook_config.py

cd /home/${UNAME}
dtach -n /tmp/ipython_notebook -Ez ipython notebook --profile=nbserver

#FIXME: dir to /home/vagrant, run as vagrant user?
echo "LD_LIBRARY_PATH=/home/${UNAME}/install/lib:/usr/local/lib:$LD_LIBRARY_PATH LD_PRELOAD=/usr/lib/libmpi.so dtach -n /tmp/ipython_notebook -Ez ipython notebook --profile=nbserver" > /etc/rc.local
echo "exit 0" >> /etc/rc.local

echo "Started IPython notebook. Point your browser to http://127.0.0.1:6868"
