.. _tasks:

🎯 Tasks
=========

.. _minidata:

Minimal Data Selection
--------------------------------------------

.. sidebar::
    Task Details
    
    :Task ID:      ``minidata``
    :Problems:     1

Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance.

**Classes**: :class:`dcbench.MiniDataProblem` :class:`dcbench.MiniDataSolution`

.. admonition:: Cloud Storage

    We recommend downloading Artifacts through the Python API, but you can also explore the Artifacts on the `Google Cloud Console <https://console.cloud.google.com/storage/browser/dcbench/minidata>`_. 


Problem Artifacts
__________________
==============  ==================================  =============================================================================
name            type                                description
==============  ==================================  =============================================================================
``train_data``  :class:`dcbench.DataPanelArtifact`  A DataPanel of train examples with columns ``id``, ``input``, and ``target``.
``test_data``   :class:`dcbench.DataPanelArtifact`  A DataPanel of test examples with columns ``id``, ``input``, and ``target``.
==============  ==================================  =============================================================================

Solution Artifacts
____________________
=============  =============================  ======================================================================
name           type                           description
=============  =============================  ======================================================================
``train_ids``  :class:`dcbench.YAMLArtifact`  A list of train example ids from the  ``id`` column of ``train_data``.
=============  =============================  ======================================================================

TODO: Provide more details on how to run miniddata evaluation.  


.. _slice_discovery:

Slice Discovery
--------------------------------------------

.. sidebar::
    Task Details
    
    :Task ID:      ``slice_discovery``
    :Problems:     4

Machine learnings models that achieve high overall accuracy often make  systematic erors on important subgroups (or *slices*) of data. When working   with high-dimensional inputs (*e.g.* images, audio) where data slices are   often unlabeled, identifying underperforming slices is challenging. In  this task, we'll develop automated slice discovery methods that mine  unstructured data for underperforming slices.

**Classes**: :class:`dcbench.SliceDiscoveryProblem` :class:`dcbench.SliceDiscoverySolution`

.. admonition:: Cloud Storage

    We recommend downloading Artifacts through the Python API, but you can also explore the Artifacts on the `Google Cloud Console <https://console.cloud.google.com/storage/browser/dcbench/slice_discovery>`_. 


Problem Artifacts
__________________
================  ======================================  =======================================================================================
name              type                                    description
================  ======================================  =======================================================================================
``predictions``   :class:`dcbench.DataPanelArtifact`      A DataPanel of the model's predictions with columns `id`,`target`, and `probs.`
``slices``        :class:`dcbench.DataPanelArtifact`      A DataPanel of the ground truth slice labels with columns  `id`, `target`, and `probs.`
``activations``   :class:`dcbench.DataPanelArtifact`      A DataPanel of the model's activations with columns `id`,`act`
``model``         :class:`dcbench.ModelArtifact`          A trained PyTorch model to audit.
``base_dataset``  :class:`dcbench.VisionDatasetArtifact`  A DataPanel representing the base dataset with columns `id` and `image`.
================  ======================================  =======================================================================================

Solution Artifacts
____________________
===============  ==================================  ==========================================================================
name             type                                description
===============  ==================================  ==========================================================================
``pred_slices``  :class:`dcbench.DataPanelArtifact`  A DataPanel of predicted slice labels with columns `id` and `pred_slices`.
===============  ==================================  ==========================================================================

TODO: Provide more details on how to run slice discovery evaluation. 


.. _miniclean:

Minimal Feature Cleaning
--------------------------------------------

.. sidebar::
    Task Details
    
    :Task ID:      ``miniclean``
    :Problems:     8

When it comes to data preparation, data cleaning is often an essential yet quite costly task. If we are given a fixed cleaning budget, the challenge is to find the training data examples that would would bring the biggest positive impact on model performance if we were to clean them.

**Classes**: :class:`dcbench.MinicleanProblem` :class:`dcbench.MiniCleanSolution`

.. admonition:: Cloud Storage

    We recommend downloading Artifacts through the Python API, but you can also explore the Artifacts on the `Google Cloud Console <https://console.cloud.google.com/storage/browser/dcbench/miniclean>`_. 


Problem Artifacts
__________________
===================  ============================  =============
name                 type                          description
===================  ============================  =============
``X_train_dirty_a``  :class:`dcbench.CSVArtifact`
``X_train_dirty_b``  :class:`dcbench.CSVArtifact`
``X_train_clean_a``  :class:`dcbench.CSVArtifact`
``X_train_clean_b``  :class:`dcbench.CSVArtifact`
``y_train_a``        :class:`dcbench.CSVArtifact`
``y_train_b``        :class:`dcbench.CSVArtifact`
``X_val``            :class:`dcbench.CSVArtifact`
``y_val``            :class:`dcbench.CSVArtifact`
``X_test``           :class:`dcbench.CSVArtifact`
``y_test``           :class:`dcbench.CSVArtifact`
===================  ============================  =============

Solution Artifacts
____________________
=============  ============================  =============
name           type                          description
=============  ============================  =============
``train_ids``  :class:`dcbench.CSVArtifact`
=============  ============================  =============

TODO: Provide more details on how to run miniclean evaluation. 
