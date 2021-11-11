.. py:currentmodule:: dcbench


💡 What is dcbench?
-------------------

This benchmark evaluates the steps in your machine learning workflow beyond model training and tuning. This includes tasks like feature cleaning, slice discovery, and coreset selection. We call these “data-centric” tasks because they're focused on exploring and manipulating data – not training models. `dcbench` supports a growing number of them:

* :any:`minidata`: Find the smallest subset of training data on which a fixed model architecture achieves accuracy above a threshold. 
* :any:`slice_discovery`: Identify subgroups on which a model underperforms.
* :any:`miniclean`: Given a fixed budget, clean input features of training data to improve model performance.  


dcbench includes tasks that look very different from one another: the inputs and
outputs of the slice discovery task are not the same as those of the
minimal data cleaning task. However, we think it important that
researchers and practitioners be able to run evaluations on data-centric
tasks across the ML lifecycle without having to learn a bunch of
different APIs or rewrite evaluation scripts.

So, dcbench is designed to be a common home for these diverse, but
related, tasks. In dcbench all of these tasks are structured in a
similar manner and they are supported by a common Python API that makes
it easy to download data, run evaluations and submit solutions.

⚙️ How it works?
---------------------------------------

``Problem``
~~~~~~~~~~~~
Each task features a collection of problems. *What is a problem?* A problem is a simply single instance of a data-centric task . For example, the Minimal Feature Cleaning task includes problems for 4 different datasets    many machine-learning workflows include a data cleaning phase where input features are audited and fixed. A potentially useful analogy is: chess problems are to a full chess game as *problems* are to the full data-centric ML lifecycle. 

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
   SliceDiscoveryProblem.describe_instances()

   # Out: TODO, get the actual dataframe output here 
   dataframe

We can get one of these problems with

.. code:: python

   problem = SliceDiscoveryProblem.instances[0]

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
