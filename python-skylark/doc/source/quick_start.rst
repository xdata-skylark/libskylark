.. highlight:: rst

Quick Start Guide
******************

Setting up a libSkylark environment is facilitated using
`Vagrant <http://www.vagrantup.com/>`_ and
`VirtualBox <https://www.virtualbox.org/>`_.
Simply follow the required steps below.

1. `Download <https://www.virtualbox.org/wiki/Downloads/>`_ and install VirtualBox.
2. `Download <http://www.vagrantup.com/downloads.html>`_ and install Vagrant.
3. Execute the following command in a terminal:

    .. code-block:: sh

        git clone https://github.com/xdata-skylark/libskylark.git
        cd libskylark/vagrant/precise64
        vagrant up
        vagrant ssh

.. note::
    The ``vagrant up`` command may take a while to bootstrap and install
    everything.


In the next section we provide more in depth information and additional
details on how the Vagrant setup works.
Additionally we provide instruction on how to install on a cluster
(see :ref:`cluster-label`) and running on AWS (see :ref:`aws-label`).

Finally, to see how the library can be used consult the
:ref:`examples-label` section.


.. _vagrant-label:

Vagrant
========

Under ``vagrant/precise64`` we provide an easy way to materialize an
64-bit Ubuntu-based virtual machine (release 12.04 LTS, aka precise) with all
software dependencies required for a fresh build of Skylark.

The ``Vagrantfile`` is the input to `Vagrant <http://www.vagrantup.com/>`_, a
program that facilitates the management of virtual machines.
Using `VirtualBox <https://www.virtualbox.org/>`_ as the backend virtualization
software (provider) we can launch a libSkylark environment indpendent of the
host operating system (*Windows*, *Mac OS X* and *Linux* host platform
require *exactly* the same commands).

*After installing VirtualBox and Vagrant* the following commands suffice

.. code-block:: sh

    cd vagrant/precise64
    vagrant up
    vagrant ssh

to get a virtual machine (VM) with all the dependencies for Skylark in place.
Then he/she can just ``ssh`` into this VM and follow the standard instructions
(see :ref:`build-libskylark-label`) for fetching, building and installing
libSkylark alone and quickly start using it.

The vagrant box can be stopped by issuing the following command:

.. code-block:: sh

   vagrant halt


.. note::

    We use *clean vagrant boxes*, automatically downloaded from the following
    urls:

    * http://cloud-images.ubuntu.com/vagrant/
    * http://www.vagrantbox.es/

.. note::

    For a Windows XP host, *PuTTy* ssh client should use the default
    connection settings: IP= ``127.0.0.1``, port= ``2222``,
    username= ``vagrant``, password= ``vagrant``.
    See also ``vagrant ssh-config`` command.


.. _cluster-label:

Cluster of vagrant-controlled VMs
==================================

Here is a simple approach for building a cluster of vagrant-controlled VMs;
it works in a local setting but since it uses the *Bridged* mode it should work
in a real cluster environment as well.
Please refer to
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

.. code-block::

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

Sketching
----------

In :file:`examples/elemental.cpp` an example is provided that illustrates
sketching.

.. literalinclude:: ../../../examples/elemental.cpp
    :language: cpp
    :linenos:


Least Square Regression
------------------------

In :file:`examples/least_squares.cpp` an example is provided that illustrates
sketching.

.. literalinclude:: ../../../examples/least_squares.cpp
    :language: cpp
    :linenos:

SVD
----

In :file:`examples/rand_svd.cpp` an example is provided that illustrates
sketching.

.. literalinclude:: ../../../examples/rand_svd.cpp
    :language: cpp
    :emphasize-lines: 79,81-82
    :linenos:

ML
---

In :file:`ml/train.cpp` an example is provided that illustrates
sketching.

.. literalinclude:: ../../../ml/train.cpp
    :language: cpp
    :linenos:

