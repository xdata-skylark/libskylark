.. highlight:: rst

Quick Start Guide
******************

There are two options for quickly installing libskylark in Linux-x86_64 systems (i.e. Linux distribution independent):

- From *conda channels* (full installation, including FFTW-based sketching)

- From *a single-file installer* (not including FFTW-based sketching)

.. _conda-channel-install:

Installing libskylark from conda channels
=========================================

Open a terminal and type (copy-paste):

.. code-block:: sh
	
	# Get miniconda for python 2.7
	curl -L -O https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
	bash Miniconda2-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda2
	
	# Get the libskylark channel
	curl -L -O https://github.com/xdata-skylark/libskylark/releases/download/v0.20/channel.tar.gz
	cd ${HOME}; tar -xvzf channel.tar.gz

	# Install fftw and libskylark conda packages
	export PATH=${HOME}/miniconda2/bin:${PATH}
	conda install fftw --yes --channel conda-forge
	conda install libskylark --yes --channel file://${HOME}/channel

Trying the installation
-----------------------

Start an ipython shell:

.. code-block:: sh

	ipython

and then input the following python snippet:

.. code-block:: python
	
	import El
	from skylark import sketch
	A = El.DistMatrix()
	El.Uniform(A, 10, 10)
	S = sketch.FJLT(10, 4) 
	SA = S * A
	print [[SA.Get(i, j) for j in range(10)] for i in range(4)]

Congratulations! You have just sketched a 10x10 distributed matrix of uniform distribution random entries into a 4x10 sketched matrix by making use of the Fast Johnson-Lindenstrauss Transform (FJLT).

Now consider hard-wiring the installation path by appending this line

.. code-block:: sh
	
	export PATH=${HOME}/miniconda2/bin:${PATH}

to your ``~/.bashrc``.

.. note::
	
	Consider expanding ``LD_LIBRARY_PATH`` if intending to use example libskylark executables: 
	``export LD_LIBRARY_PATH=${HOME}/miniconda2/lib64:${HOME}/miniconda2/lib:${LD_LIBRARY_PATH}"``


.. _single-file-installer:

Installing libskylark from a single-file installer
==================================================

Open a terminal and type (copy-paste):

.. code-block:: sh

	# Download the installer and install libskylark
	curl -L -O https://github.com/xdata-skylark/libskylark/releases/download/v0.20/install.sh
	bash install.sh -b -p ${HOME}/libskylark
	export PATH=${HOME}/libskylark/bin:${PATH}

Trying the installation
-----------------------

Start an ipython shell:

.. code-block:: sh
	
	ipython

and then input the following python snippet:

.. code-block:: python
	
	import El
	from skylark import sketch
	A = El.Matrix()
	El.Uniform(A, 10, 10)
	S = sketch.JLT(10, 4) 
	SA = S * A
	print SA.ToNumPy()

Congratulations! You have just sketched a 10x10 matrix of uniform distribution random entries into a 4x10 sketched matrix by making use of the Johnson-Lindenstrauss Transform (JLT).

Now consider hard-wiring the installation path by appending this line

.. code-block:: sh

	export PATH=${HOME}/libskylark/bin:${PATH}

to your ``~/.bashrc``.

.. note::
	
	Consider expanding ``LD_LIBRARY_PATH`` if intending to use example libskylark executables: 
	``export LD_LIBRARY_PATH=${HOME}/miniconda2/lib64:${HOME}/miniconda2/lib:${LD_LIBRARY_PATH}"``


.. _vagrant-label:


Vagrant
========

Setting up a libSkylark environment is facilitated using
`Vagrant <http://www.vagrantup.com/>`_ and
`VirtualBox <https://www.virtualbox.org/>`_.

.. note:: Make sure to use Vagrant >= 1.6.3! The configuration file will
    not be compatible with older versions.

Simply follow the instructions shown below.

1. Install VirtualBox (`download VirtualBox <https://www.virtualbox.org/wiki/Downloads/>`_)
2. Install Vagrant (`download Vagrant <http://www.vagrantup.com/downloads.html>`_).
3. Execute the following commands in a terminal:

    .. code-block:: sh

        git clone https://github.com/xdata-skylark/libskylark.git
        cd libskylark/vagrant/trusty64
        vagrant up
        vagrant ssh

.. note:: The ``vagrant up`` command is downloading, building and installing
    libsklylark and its dependencies. This step will most likely take a while
    to complete.

To exit the virtual environment type ``exit`` in the shell followed by
``vagrant halt`` in order to power down the virtual machine.
In order to reconnect to the virtual machine, either use the VirtualBox
interface to start the virtual machine and then login with the
``vagrant/vagrant`` user/password or execute

.. code-block:: sh

    vagrant up --no-provision
    vagrant ssh


.. note:: Depending on your Vagrant version, omitting the ``--no-provision``
    argument (after a ``vagrant halt``) is considerably slower because the
    ``vagrant up`` reruns parts of the bootstrap script.

.. note:: The following environment variables are available after login,
    pointing to the libSkylark source and install directory:

    .. code-block:: sh

        SKYLARK_SRC_DIR=/home/vagrant/libskylark
        SKYLARK_INSTALL_DIR=/home/vagrant/install


In the next section we provide more in depth details on how the Vagrant setup
works and some trouble shooting.
Additionally we provide instruction on how to install on a cluster
(see :ref:`cluster-label`) and running on AWS (see :ref:`aws-label`).

Finally, some examples are provided in the :ref:`examples-label` section.

.. note::

    You might get a warning concerning AVX instructions when running the
    examples

        `OpenBLAS : Your OS does not support AVX instructions. OpenBLAS is using Nehalem kernels as a fallback, which may give poorer performance.`

    because some VM providers do not support AVX instructions yet (e.g. see:
    `https://www.virtualbox.org/ticket/11347`).



Under ``vagrant/trusty64`` we provide an easy way to materialize an
64-bit Ubuntu-based virtual machine (release 14.04 LTS, aka Trusty Tahr) with all
software dependencies required for a fresh build of libSkylark.

The ``Vagrantfile`` is the input to `Vagrant <http://www.vagrantup.com/>`_, a
program that facilitates the management of virtual machines.
Using `VirtualBox <https://www.virtualbox.org/>`_ as the backend virtualization
software (provider) we can launch a libSkylark environment independent of the
host operating system (*Windows*, *Mac OS X* and *Linux* host platform
require *exactly* the same commands).

*After installing VirtualBox and Vagrant* the following commands suffice

.. code-block:: sh

    git clone https://github.com/xdata-skylark/libskylark.git
    cd vagrant/trusty64
    vagrant up
    vagrant ssh

to get a virtual machine (VM) with libSkylark and all its dependencies.
If you want to customize the installed libSkylark follow the instructions
(see :ref:`build-libskylark-label`) for building and installing
libSkylark.

.. note::

    To customize the amount of memory (4096) and CPU (75%) cap that will be
    used for the VM, check the Vagrant file.

The vagrant box can be stopped by issuing ``vagrant halt``.


.. note::

    We use *clean vagrant boxes*, automatically downloaded from the following
    urls:

    * http://cloud-images.ubuntu.com/vagrant/
    * http://www.vagrantbox.es/

.. note::

    For a Windows XP host, *PuTTy* ssh client should use the default
    connection settings: IP= ``127.0.0.1``, port= ``2222``,
    username= ``vagrant``, password= ``vagrant``.

.. hint::

    If you get a timeout error (waiting for the virtual machine to boot)

    .. code-block:: sh

        [...]
        default: Warning: Connection timeout. Retrying...
        default: Warning: Connection timeout. Retrying...
        Timed out while waiting for the machine to boot. This means that
        Vagrant was unable to communicate with the guest machine within
        the configured ("config.vm.boot_timeout" value) time period.

    during ``vagrant up``, halt the virtual machine and ``vagrant up`` again.

.. hint::

    In case Vagrant crashes in the bootstrap phase, the safest thing is to
    delete the virtual machine (halt than delete the machine in the VirtualBox
    GUI) and start with a fresh Vagrant build.


Command-line Usage
==================

While libSkylark is designed to be a library, it also provides access to many
of its feature using standalone applications. For all applications, running with
the --help options reveals the command-line options.

The following example sessions assume the following commands have been performed
in advance (it downloads and extracts some demo data):

.. code-block:: sh

        wget  https://www.dropbox.com/s/0ilbyeem2ihmvzw/data.tar.gz
        tar -xvzf data.tar.gz


.. _svd_app:

Approximate Singular Value Decomposition
----------------------------------------

Building libSkylark creates an executable called ``skylark_svd`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode
to compute an approximate SVD with 10 leading singular vectors as follows:

.. code-block:: sh

         skylark_svd -k 10 --prefix usps data/usps.train

The files usps.U.txt, usps.S.txt and usps.V.txt contain the approximate SVD that
was computed.


Linear Least-Squares
--------------------

Building libSkylark creates an executable called ``skylark_linear`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode to
approximately solve linear least squares problems. Here is an example command-line:

.. code-block:: sh

         skylark_linear data/cpu.train cpu.sol

The file cpu.sol.txt contains the computed approximate solution.

.. note::

   Currently, only overdetermined least squares of dense matrices is supported.
   No regularization options are available. This will be relaxed in the future.

.. _ml_example:

Learning Non-Linear Models
--------------------------

libSkylark offers two mechanisms to learn non-linear models: ADMM-based solver
and kernel ridge regression solver.

Kernel Ridge Regression solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In :file:`${SKYLARK_SRC_DIR}/ml/skylark_krr.cpp`, kernel ridge regression is
setup to train and predict with a  kernel based model for
nonlinear classification and regression problems. The code incorporates various
algorithms, with varying trade-offs in speed, memory and accuracy.

Building libSkylark creates an executable called ``skylark_krrl`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode as follows. 

1. Train a classifier using the Gaussian Kernel

.. code-block:: sh

        mpiexec -np 4 ./skylark_krr -a 1 -k 0 -g 10 -f 1000 --model model data/usps.train 

2. Test accuracy of the generated model and generate predictions

.. code-block:: sh

        mpiexec -np 4 ./skylark_krr --outputfile output --model model usps.test 

ADMM-based solver
^^^^^^^^^^^^^^^^^
In :file:`${SKYLARK_SRC_DIR}/ml/skylark_ml.cpp`, an ADMM-based solver is
setup to train and predict with a randomized kernel based model for
nonlinear classification and regression problems.

Building libSkylark creates an executable called ``skylark_ml`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode as follows.

1. Train an SVM with Randomized Gaussian Kernel

.. code-block:: sh

        mpiexec -np 4 ./skylark_ml -g 10 -k 1 -l 2 -i 30 -f 1000 --trainfile data/usps.train --valfile data/usps.test --modelfile model

2. Test accuracy of the generated model and generate predictions

.. code-block:: sh

        mpiexec -np 4 ./skylark_ml --outputfile output --testfile data/usps.test --modelfile model

An output file named output.txt with the predicted labels is created.

3. In the above, the entire test data is loaded to memory and the result is computed. This is the fast, but it is limited to cases where the test data is fixed. skylark_ml also supports an interactive (streaming) mode in which the software prompts for data line by line from stdin and outputs in prediction after each line is received. The mode is invoked by not passing either training data or testing data. An example for using this mode is the following:

.. code-block:: sh

         cut -d ' ' -f 1 --complement data/usps.test | skylark_ml --modelfile model

.. note::

	Interactive mode is much slower, since there is significant overhead that is repeated for each new line of data. In the future, batching will be supported, to avoid some of this overhead.

5. It is also possible to load the model in Python and do predictions. Here is an example:

.. code-block:: python

         import skylark.io, skylark.ml.modeling
         model = skylark.ml.modeling.LinearizedKernelModel('model')
         testfile = skylark.io.libsvm('data/usps.test')
         Xt, Yt = testfile.read(model.get_input_dimension())
         Yp = model.predict(Xt)

Community Detection Using Seed Node
-----------------------------------

Building libSkylark creates an executable called ``skylark_community`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode as follows.
(Note that an interactive mode is also present.)

1. Prepare the input. Basically, the graph is described in a text file, with each
row an edge. Each row has two strings that identify the node. The identifier can be
arbitrary (does not have to be numeric). For example:

.. code-block:: sh

         cat data/two_triangles
         A1 A2
         A2 A3
         A1 A3
         B1 B2
         B2 B3
         B1 B3
         A1 B1

2. Detect the community with A1 as the seed:

.. code-block:: sh

         skylark_community --graphfile data/two_triangles --seed A1

3. The community detection algorithm is local, so it operates only on a few nodes.
So, for large graphs the majority of the time for detecting a single community will
be spent on just loading the file. To detect multiple communities with the graph
already loaded in memory, an interactive quiet mode is provided. Here is an
example:

.. code-block:: sh

         cat data/seeds | skylark_community --graphfile data/two_triangles -i -q

Approximate Adjacency Spectral Embedding
----------------------------------------

Building libSkylark creates an executable called ``skylark_graph_se`` in
``$SKYLARK_INSTALL_DIR/bin``. This executable can be used in standalone mode as follows.

1. Prepare the input. Basically, the graph is described in a text file, with each
row an edge. Each row has two strings that identify the node. The identifier can be
arbitrary (does not have to be numeric). For example:

.. code-block:: sh

         cat data/two_triangles
         A1 A2
         A2 A3
         A1 A3
         B1 B2
         B2 B3
         B1 B3
         A1 B1

2. Compute a two dimensional embedding.

.. code-block:: sh

         skylark_graph_se -k 2 data/two_triangles

3. Two files are created: out.index.txt and out.vec.txt. The vec file has the embedding
vectors: each row corresponds to an index. The index file maps row index to vertices. 
The prefix can be called using the prefix command line option (default is `out`).

.. _cluster-label:

Cluster of vagrant-controlled VMs
==================================

Here is a simple approach for building a cluster of vagrant-controlled VMs;
it works in a local setting but since it uses the *bridged* mode it should work
in a real cluster environment as well. Please refer to
`this blog entry <https://blogs.oracle.com/fatbloke/entry/networking_in_virtualbox1>`_
for more information on `Networking in VirtualBox`.

* In ``Vagrantfile`` set:

.. code-block:: sh

   config.vm.network :public_network

* During ``vagrant up`` choose the ethernet interface to use.

* After ``vagrant ssh`` do ``ifconfig``, get the interface name - say
    ``eth1`` - and then for the VMs at nodes 1, 2,...  do:

.. code-block:: sh

   sudo /sbin/ifconfig eth1:vm 192.168.100.1
   sudo /sbin/ifconfig eth1:vm 192.168.100.2
   ...

* ``192.168.100.xxx`` will be the names to put in the "hosts" file for MPI
    daemons to use; as previously noted: ``vagrant``/``vagrant`` is the default
    user/password combination for ssh.


.. _aws-label:

Running Vagrant on AWS
=======================

This follows the instructions found at:

* https://github.com/mitchellh/vagrant-aws
* http://www.devopsdiary.com/blog/2013/05/07/automated-deployment-of-aws-ec2-instances-with-vagrant-and-puppet/

Using the provided ``bootstrap.sh`` file in combination with the ``aws``
plugin (see link above) one can deploy on AWS. First adapt the Vagrant file as
shown below:

.. code-block:: sh

    Vagrant.configure("2") do |config|
        config.vm.box = "skylark"
        config.vm.provision :shell, :path => "bootstrap.sh"

        config.vm.provider :aws do |aws, override|
            aws.access_key_id     = "XYZ"
            aws.secret_access_key = "AAA"
            aws.keypair_name      = "keynam"
            aws.security_groups   = ["ssh-rule"]

            aws.ami           = "ami-fa9cf1ca"
            aws.region        = "us-west-2"
            aws.instance_type = "t1.micro"

            override.ssh.username         = "ubuntu"
            override.ssh.private_key_path = "aws.pem"
        end
    end


Then execute:

* ``vagrant plugin install vagrant-aws``
* ``vagrant box add skylark https://github.com/mitchellh/vagrant-aws/raw/master/dummy.box``
* ``vagrant up --provider=aws``
* ``vagrant ssh``

It will take a while to compile and install everything specified in the
``bootstrap.sh`` script.


Running MPI on Hadoop/Yarn
==========================

libSkylark can be run on on a Yarn scheduled cluster using the blueprint
outlined below.

.. note::

    The installation requires admin privileges. Also make sure to satisfy the
    recommended dependencies.


1. Building MPICH2
------------------

Download the MPICH version specified in the dependencies (MPICH-3.1.2).

.. code-block:: sh

    sh autogen.sh
    ./configure --prefix=$HOME/mpi/bin
    make
    make install
    sudo ln -s /home/ubuntu/mpi/bin/bin/* /usr/local/bin


and copy MPI binaries to all nodes, i.e.

.. code-block:: sh

    scp /home/ubuntu/mpi/bin/bin/* root@hadoopX:/usr/local/bin

Make sure that all the MPI executables are accessible on all the nodes.


2. Building mpich2-yarn
-----------------------

Finally we have all the dependencies ready and can continue to mpich2-yarn
https://github.com/alibaba/mpich2-yarn:

.. code-block:: sh

    cd ~/mpi
    git clone https://github.com/alibaba/mpich2-yarn
    cd mpich2-yarn
    mvn clean package -Dmaven.test.skip=true


The configuration for Hadoop/Yarn can be found on on the Github project page.
Note that the HDFS permission must be set correctly for the temporary folders
used to distribute MPI executables.


Finally, an MPI job can be submitted, i.e. with

.. code-block:: sh

    sudo -u yarn hadoop jar target/mpich2-yarn-1.0-SNAPSHOT.jar -a hellow -M 1024 -m 1024 -n 1

where we chose to run as the yarn user (double check permission for any user
that runs `mpich2-yarn`).
More examples can be found on the mpich2-yarn Github page.



.. _examples-label:

Examples of Library Usage
=========================

Here, we provide a flavor of the library and its usage. We assume that
the reader is familiar with sketching and its applications in randomized
Numerical Linear algebra and Machine Learning. If not, we refer the
reader to subsequent sections which provide the necessary background and
references.

When libSkylark is built, executables instantiating these examples can be found
under ``$SKYLARK_INSTALL_DIR/bin/skylark_examples`` and
``$SKYLARK_INSTALL_DIR/bin``.

In addition Vagrant provisions a
`IPyhton notebook <http://ipython.org/notebook.html>` server with access to the
libSkylark Python bindings. To play with libSkylark just open your browser and
direct it to `localhost:6868`.


Sketching
----------

In :file:`${SKYLARK_SRC_DIR}/examples/elemental.cpp` an example is provided that illustrates
sketching. Below, three types of sketches are illustrated in the highlighted lines:

1. a 1D row-distributed `Elemental <http://libelemental.org>`_ dense matrix is sketched using the `JLT (Johnson-Lindenstrauss) Transform` into a local matrix
2. a row-distributed `Elemental <http://libelemental.org>`_ dense matrix is sketched using  `FJLT (Fast Johnson-Lindenstrauss transform)` involving faster FFT-like operations.
3. a 2D block-cyclic distributed `Combinatorial BLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ sparse matrix is sketched to a dense local Elemental matrix using `Clarkson-Woodruff` transform which involves hashing the rows.

.. literalinclude:: ../../examples/elemental.cpp
    :language: cpp
    :emphasize-lines: 65,66,69,73,96,97,100,103,118,119,122,126
    :linenos:


For sketching Elemental and CombBLAS matrices via Python bindings, run, for example, the following:

.. code-block:: sh

	mpiexec -np 4 python $SKYLARK_SRC_DIR/python-skylark/skylark/examples/example_sketch.py
  	mpiexec -np 4 python $SKYLARK_SRC_DIR/python-skylark/skylark/examples/example_sparse_sketch.py

.. note::

	Currently, we have stable python bindings only for sketching layer. This too might change with newer versions of Elemental.

Fast Least Square Regression
-----------------------------

In :file:`${SKYLARK_SRC_DIR}/examples/least_squares.cpp` examples are provided on how least
squares problems can be solved faster using sketching. One approach is
of the flavor of `sketch-and-solve` geared towards lower-precision
solutions, while the other approach uses sketching to construct a
preconditioner to obtain high-precision solutions faster.

.. literalinclude:: ../../examples/least_squares.cpp
    :language: cpp
    :emphasize-lines: 88,102
    :linenos:

For an example of `sketch-and-solve` regression in Python, run:

.. code-block:: sh

        wget http://vikas.sindhwani.org/data.tar.gz
        tar -xvzf data.tar.gz
	mpiexec -np 4 python $SKYLARK_SRC_DIR/python-skylark/skylark/examples/sketch_solve.py data/usps.train data/usps.test

SVD
----

In :file:`${SKYLARK_SRC_DIR}/nla/skylark_svd.cpp` an example is provided that illustrates randomized singular value decompositions.

.. literalinclude:: ../../nla/skylark_svd.cpp
    :language: cpp
    :emphasize-lines: 40
    :linenos:
