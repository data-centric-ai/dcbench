.. py:currentmodule:: dcbench




🧭 API Walkthrough
---------------------------------------

``Task``
~~~~~~~~~~~~
``dcbench`` supports a diverse set of data-centric tasks (*e.g.* :any:`slice_discovery`). 
You can explore the supported tasks in the documentation (:any:`tasks`) or via the Python API:

.. ipython:: python

   import dcbench
   dcbench.tasks


In the ``dcbench`` API, each task is represented by a :class:`dcbench.Task` object that can be accessed by *task_id* (*e.g.* ``dcbench.tasks["slice_discovery"]``). These task objects hold metadata about the task and hold pointers to task-specific :class:`dcbench.Problem` and :class:`dcbench.Solution` subclasses, discussed below.  

``Problem``
~~~~~~~~~~~~
Each task features a collection of *problems* (*i.e.* instances of the task). For example, the :any:`slice_discovery` task includes hundreds of problems across a number of different datasets. We can explore a task's problems in ``dcbench``:  

.. ipython:: python
   slice_discovery = dcbench.tasks["slice_discovery"]
   slice_discovery.problems

All of a task's problems share the same structure and use the same evaluation scripts.
This is specified via task-specific subclasses of :class:`dcbench.Problem` (*e.g.* :class:`~dcbench.SliceDiscoveryProblem`). The problems themselves are instances of these subclasses. We can access a problem using it's id:

.. ipython:: python

   problem = slice_discovery.problems["p_72063"]
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

Note that :class:`Artifact` objects don't actually hold their underlying data in memory. Instead, they hold pointers to where the :class:`Artifact` lives in `dcbench cloud storage <https://console.cloud.google.com/storage/browser/dcbench?authuser=1&project=hai-gcp-fine-grained&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false>`_ and, if it's been downloaded,  where it lives locally on disk. This makes the :class:`Problem` objects very lightweight.  


.. admonition:: Downloading to Disk

   By default, ``dcbench`` downloads artifacts to ``~/.dcbench/artifacts`` but this can be configured in the ``dcbench`` settings TODO: add support for configuration. To download an :class:`Artifact`  via the Python API, use :meth:`Artifact.download()`. You can also download all the artifacts in a problem with :class:`Problem.download()`.

**Loading into memory.** `dcbench` includes loading functionality for each artifact type. To load an artifact into memory you can use `artifact.load()` . Note that this will also download the artifact to disk if it hasn't yet been downloaded. 


``Solution``
~~~~~~~~~~~~