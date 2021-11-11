.. py:currentmodule:: dcbench


💡 What is dcbench?
-------------------

This is a benchmark for evaluating the steps in your machine learning workflow beyond model training and tuning. This includes tasks like label correction, slice discovery, and coreset selection. We call these “data-centric” tasks and dcbench supports a growing number of them:

* Minimal data cleaning (``miniclean``) 
* Label Correction (``labelfix``) 
* Slice Discovery (``slice-discovery``) 
* Minimal training dataset selection (``minitrain``)

Unlike machine learning benchmarks focused on model training, dcbench
includes tasks that look very different from one another: the inputs and
outputs of the slice discovery task are not the same as those of the
minimal data cleaning task. However, we think it important that
researchers and practitioners be able to run evaluations on data-centric
tasks across the ML lifecycle without having to learn a bunch of
different APIs or rewrite evaluation scripts.

So, dcbench is designed to be a common home for these diverse, but
connected, tasks. In dcbench all of these tasks are structured in a
similar manner and they are supported by a common Python API that makes
it easy to download data, run evaluations and submit solutions.

⚙️ How it works?
---------------------------------------

``Problem``
~~~~~~~~~~~~
Each task features a a collection of *problems*. *What is a problem?* A useful analogy is: chess problems are to a full chess game as *problems* are to the full data-centric ML lifecycle. For example, many machine-learning workflows include a label correction phase where labels are audited and fixed. Our benchmark includes a collection of label cleaning *problems* each with a different dataset and set of sullied labels to be cleaned.

The benchmark supports a diverse set of problems that may look very different from one another. For example, a slice discovery problem has different inputs and outputs than a data cleaning problem. To deal with this, we group problems by *task.* In ``dcbench``, each task is represented by a subclass of :class:`dcbench.Problem` (*e.g.*
:class:`dcbench.SliceDiscoveryProblem`, :class:`dcbench.MinicleanProblem`). The problems
themselves are represented by instances of these subclasses.

We can get a list all of the problem classes in ``dcbench`` with:

.. code:: python

   import dcbench
   dcbench.tasks

   # OUT: 
   [<class 'dcbench.tasks.miniclean.problem.MinicleanProblem'>, <class 'dcbench.tasks.slice.SliceDiscoveryProblem'>]

``dcbench`` includes a set of problems for each task. We can list them
with:

.. code:: python

   from dcbench import SliceDiscoveryProblem
   SliceDiscoveryProblem.instances

   # Out: TODO, get the actual dataframe output here 
   dataframe

We can get one of these problems with

.. code:: python

   problem = SliceDiscoveryProblem.from_id("eda4")

``Artifact``
~~~~~~~~~~~~

Each *problem* is made up of a set of artifacts: a dataset with labelsto clean, a dataset and a model to perform error analysis on. In ``dcbench`` , these artifacts are represented by instances of
:class:`dcbench.Artifact`. We can think of each :class`Problem` object as a container for :class:`Artifact` objects. 

.. code:: python

   problem.artifacts

   # Out: 
   {
      "dataset": CSVArtifact()
   }

   artifact: CSVArtifact = problem["dataset"]


Note that :class:`Artifact` objects don't actually hold their underlying data in memory. Instead, they hold pointers to where the :class:`Artifact` lives in `dcbench cloud storage <https://console.cloud.google.com/storage/browser/dcbench?authuser=1&project=hai-gcp-fine-grained&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false>`_ and, if it's been downloaded,  where it lives locally on disk. This makes the :class:`Problem` objects very lightweight.  

**Downloading to disk.** By default, `dcbench` downloads artifacts to `~/.dcbench/artifacts` but this can be configured in the dcbench settings TODO: add support for configuration. To download an :class:`Artifact`  via the Python API, use :meth:`Artifact.download()`. You can also download all the artifacts in a problem with :class:`Problem.download()`.

**Loading into memory.** `dcbench` includes loading functionality for each artifact type. To load an artifact into memory you can use `artifact.load()` . Note that this will also download the artifact if it hasn't yet been downloaded. 

Finally,  we should point out that `problem` is a Python mapping, so we can index it directly to load artifacts.  

.. code:: python

   # this is equivalent to problem.artifacts["dataset"].load()
   df: pd.DataFrame = problem["dataset"] 
