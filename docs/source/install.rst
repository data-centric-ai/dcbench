
.. _installing:

🚀 Installing dcbench
============================

This section describes how to install the ``dcbench`` Python package.

.. code-block:: bash

    pip install dcbench

.. admonition:: Optional

    Some parts of ``dcbench`` rely on optional dependencies. If you know which optional dependencies you'd like to install, you can do so using something like ``pip install dcbench[dev]`` instead. See ``setup.py`` for a full list of optional dependencies.

Installing from branch
-----------------------

To install from a specific branch use the command below, replacing ``main`` with the name of `any branch in the dcbench repository <https://github.com/data-centric-ai/dcbench/branches>`_.

.. code-block:: bash

    pip install "dcbench @ git+https://github.com/data-centric-ai/dcbench@main"

 
Installing from clone
-----------------------
You can install from a clone of the ``dcbench`` `repo <https://github.com/data-centric-ai/dcbench/branches>`_ with: 

.. code-block:: bash

    git clone https://github.com/data-centric-ai/dcbench.git
    cd dcbench
    pip install -e .

.. _configuring:

⚙️ Configuring dcbench
============================

Several aspects of ``dcbench`` behavior can be configured by the user. 
For example, one may wish to change the directory in which ``dcbench`` downloads artifacts (by default this is ``~/.dcbench``).

You can see the current state of the ``dcbench`` configuration with:

.. ipython:: python

    import dcbench
    dcbench.config    

Configuring with YAML
----------------------

To change the configuration create a YAML file, like the one below:

.. code-block:: yaml
    local_dir: "/path/to/storage"
    public_bucket_name: "dcbench-test"

Then set the environment variable ``DCBENCH_CONFIG`` to point to the file:

.. code-block:: bash

    export DCBENCH_CONFIG="/path/to/dcbench-config.yaml"

If you're using a conda, you can permanently set this variable for your environment:

.. code-block:: bash

    conda env config vars set DCBENCH_CONFIG="path/to/dcbench-config.yaml"
    conda activate env_name  # need to reactivate the environment 


Configuring Programmatically
------------------------------

You can also update the config programmatically, though unlike the YAML method above, these changes will not persist beyond the lifetime of your program. 

.. code-block:: python

    dcbench.config.local_dir = "/path/to/storage"
    dcbench.config.public_bucket_name = "dcbench-test"

