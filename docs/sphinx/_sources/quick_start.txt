.. highlight:: rst

Quick Start Guide
******************

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
        cd libskylark/vagrant/precise64
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
    pointing to the libskylark source and install directory:

    .. code-block:: sh

        SKYLARK_SRC_DIR=/home/vagrant/libskylark
        SKYLARK_INSTALL_DIR=/home/vagrant/install


In the next section we provide more in depth details on how the Vagrant setup
works and some trouble shooting.
Additionally we provide instruction on how to install on a cluster
(see :ref:`cluster-label`) and running on AWS (see :ref:`aws-label`).

Finally, some examples are provided in the :ref:`examples-label` section.


.. _vagrant-label:

Vagrant
========

Under ``vagrant/precise64`` we provide an easy way to materialize an
64-bit Ubuntu-based virtual machine (release 12.04 LTS, aka precise) with all
software dependencies required for a fresh build of Skylark.

The ``Vagrantfile`` is the input to `Vagrant <http://www.vagrantup.com/>`_, a
program that facilitates the management of virtual machines.
Using `VirtualBox <https://www.virtualbox.org/>`_ as the backend virtualization
software (provider) we can launch a libSkylark environment independent of the
host operating system (*Windows*, *Mac OS X* and *Linux* host platform
require *exactly* the same commands).

*After installing VirtualBox and Vagrant* the following commands suffice

.. code-block:: sh

    git clone https://github.com/xdata-skylark/libskylark.git
    cd vagrant/precise64
    vagrant up
    vagrant ssh

to get a virtual machine (VM) with libSkylark and all its dependencies.
If you want to customize the installed libSkylark follow the instructions
(see :ref:`build-libskylark-label`) for building and installing
libSkylark.

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

* After ``vagrant ssh`` do ``ifconfig``, get the interface name - let's say
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


.. _examples-label:

Examples
=========

Here, we provide a flavor of the library and its usage. We assume that
the reader is familiar with sketching and its applications in randomized
Numerical Linear algebra and Machine Learning. If not, we refer the
reader to subsequent sections which provide the necessary background and
references.

When libskylark is built, executables instantiating these examples can be found
under ``$SKYLARK_INSTALL_DIR/bin/examples`` and
``$SKYLARK_INSTALL_DIR/bin/ml``.

Sketching
----------

In :file:`${SKYLARK_SRC_DIR}/examples/elemental.cpp` an example is provided that illustrates
sketching. Below, three types of sketches are illustrated in the highlighted lines:

1. a 1D row-distributed `Elemental <http://libelemental.org>`_ dense matrix is sketched using the `JLT (Johnson-Lindenstrauss) Transform` into a local matrix
2. a row-distributed `Elemental <http://libelemental.org>`_ dense matrix is sketched using  `FJLT (Fast Johnson-Lindenstrauss transform)` involving faster FFT-like operations.
3. a 2D block-cyclic distributed `Combinatorial BLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ sparse matrix is sketched to a dense local Elemental matrix using `Clarkson-Woodruff` transform which involves hashing the rows.

.. literalinclude:: ../../../examples/elemental.cpp
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

.. literalinclude:: ../../../examples/least_squares.cpp
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

In :file:`${SKYLARK_SRC_DIR}/examples/rand_svd.cpp` an example is provided that illustrates randomized singular value decompositions.

.. literalinclude:: ../../../examples/rand_svd.cpp
    :language: cpp
    :emphasize-lines: 79,81-82
    :linenos:

.. _ml_example:

ML
---

In :file:`${SKYLARK_SRC_DIR}/ml/train.cpp` and :file:`${SKYLARK_SRC_DIR}/ml/run.hpp`, an ADMM-based solver is
setup to train and predict with a randomized kernel based model for
nonlinear classification and regression problems.

Building libskylark creates an executable called ``skylark_ml`` in
``$SKYLARK_INSTALL_DIR/bin/ml`` under the libskylark installation folder.
This executable can be used in standalone mode as follows.

1. Download USPS digit recognition dataset (in various supported formats).

.. code-block:: sh

        wget http://vikas.sindhwani.org/data.tar.gz
        tar -xvzf data.tar.gz

The supported fileformats are described in :ref:`ml_io`.

2. Train an SVM with Randomized Gaussian Kernel

.. code-block:: sh

        mpiexec -np 4 ./skylark_ml -g 10 -k 1 -l 2 -i 20 --trainfile data/usps.train --valfile data/usps.test --modelfile model

3. Test accuracy of the generated model

.. code-block:: sh

        mpiexec -np 4 ./skylark_ml --testfile data/usps.test --modelfile model

.. note::

	In testing mode, the entire test data is currently loaded in memory while ideally the model should be applied in a streaming fashion. A separate file containing predictions is currently not generated. These restrictions will be relaxed.
