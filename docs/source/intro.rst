💡 What is dcbench?
-------------------

This benchmark evaluates the steps in your machine learning workflow beyond model training and tuning. This includes feature cleaning, slice discovery, and coreset selection. We call these “data-centric” tasks because they're focused on exploring and manipulating data – not training models. ``dcbench`` supports a growing number of them:

* :any:`minidata`: Find the smallest subset of training data on which a fixed model architecture achieves accuracy above a threshold. 
* :any:`slice_discovery`: Identify subgroups on which a model underperforms.
* :any:`budgetclean`: Given a fixed budget, clean input features of training data to improve model performance.  


``dcbench`` includes tasks that look very different from one another: the inputs and
outputs of the slice discovery task are not the same as those of the
minimal data cleaning task. However, we think it important that
researchers and practitioners be able to run evaluations on data-centric
tasks across the ML lifecycle without having to learn a bunch of
different APIs or rewrite evaluation scripts.

So, ``dcbench`` is designed to be a common home for these diverse, but
related, tasks. In ``dcbench`` all of these tasks are structured in a
similar manner and they are supported by a common Python API that makes
it easy to download data, run evaluations, and compare methods.

.. py:currentmodule:: dcbench




🧭 API Walkthrough
---------------------------------------
..
    TODO: Add a schematic outlining the clas structure 

.. code-block:: bash

    pip install dcbench

.. _task-intro:


``Task``
~~~~~~~~~~~~
``dcbench`` supports a diverse set of data-centric tasks (*e.g.* :any:`slice_discovery`). 
You can explore the supported tasks in the documentation (:any:`tasks`) or via the Python API:

.. ipython:: python

   import dcbench
   dcbench.tasks


In the ``dcbench`` API, each task is represented by a :class:`dcbench.Task` object that can be accessed by *task_id* (*e.g.* ``dcbench.slice_discovery``). These task objects hold metadata about the task and hold pointers to task-specific :class:`dcbench.Problem` and :class:`dcbench.Solution` subclasses, discussed below.  

.. _problem-intro:

``Problem``
~~~~~~~~~~~~
Each task features a collection of *problems* (*i.e.* instances of the task). For example, the :any:`slice_discovery` task includes hundreds of problems across a number of different datasets. We can explore a task's problems in ``dcbench``:  

.. ipython:: python

   dcbench.tasks["slice_discovery"].problems

All of a task's problems share the same structure and use the same evaluation scripts.
This is specified via task-specific subclasses of :class:`dcbench.Problem` (*e.g.* :class:`~dcbench.SliceDiscoveryProblem`). The problems themselves are instances of these subclasses. We can access a  problem using it's id:

.. ipython:: python

   problem = dcbench.tasks["slice_discovery"].problems["p_118919"]
   problem


``Artifact``
~~~~~~~~~~~~

Each *problem* is made up of a set of artifacts: a dataset with features to clean, a dataset and a model to perform error analysis on. In ``dcbench`` , these artifacts are represented by instances of
:class:`dcbench.Artifact`. We can think of each :class:`Problem` object as a container for :class:`Artifact` objects. 

.. ipython:: python

   problem.artifacts

Note that :class:`~dcbench.Artifact` objects don't actually hold their underlying data in memory. Instead, they hold pointers to where the :class:`Artifact` lives in ``dcbench`` `cloud storage <https://console.cloud.google.com/storage/browser/dcbench?authuser=1&project=hai-gcp-fine-grained&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false>`_ and, if it's been downloaded, where it lives locally on disk. This makes the :class:`Problem` objects very lightweight.  

``dcbench`` includes loading functionality for each artifact type. To load an artifact into memory we can use :meth:`~dcbench.Artifact.load()` . Note that this will also download the artifact to disk if it hasn't yet been downloaded. 

.. ipython:: python
   
   problem.artifacts["model"]

Easier yet, we can use the index operator directly on :class:`Problem` objects to both fetch the artifact and load it into memory. 

.. ipython:: python
   
   problem["activations"]  # shorthand for problem.artifacts["model"].load()


.. admonition:: Downloading to Disk

   By default, ``dcbench`` downloads artifacts to ``~/.dcbench`` but this can be configured by creating a ``dcbench-config.yaml`` as described in :any:`configuring`. To download an :class:`Artifact`  via the Python API, use :meth:`Artifact.download()`. You can also download all the artifacts in a problem with :class:`Problem.download()`.


``Solution``
~~~~~~~~~~~~