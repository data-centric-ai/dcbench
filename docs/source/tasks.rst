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
==============  ==================================  ==================================================================================
name            type                                description
==============  ==================================  ==================================================================================
``train_data``  :class:`dcbench.DataPanelArtifact`  A DataPanel of train examples with columns ``id``, ``input``, and ``target``.
``val_data``    :class:`dcbench.DataPanelArtifact`  A DataPanel of validation examples with columns ``id``, ``input``, and ``target``.
``test_data``   :class:`dcbench.DataPanelArtifact`  A DataPanel of test examples with columns ``id``, ``input``, and ``target``.
==============  ==================================  ==================================================================================

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


.. _budgetclean:

Minimal Feature Cleaning
--------------------------------------------

.. sidebar::
    Task Details
    
    :Task ID:      ``budgetclean``
    :Problems:     144

When it comes to data preparation, data cleaning is often an essential yet quite costly task. If we are given a fixed cleaning budget, the challenge is to find the training data examples that would would bring the biggest positive impact on model performance if we were to clean them.

**Classes**: :class:`dcbench.BudgetcleanProblem` :class:`dcbench.BudgetcleanSolution`

.. admonition:: Cloud Storage

    We recommend downloading Artifacts through the Python API, but you can also explore the Artifacts on the `Google Cloud Console <https://console.cloud.google.com/storage/browser/dcbench/budgetclean>`_. 


Problem Artifacts
__________________
=================  ============================  ========================================================================================================================================
name               type                          description
=================  ============================  ========================================================================================================================================
``X_train_dirty``  :class:`dcbench.CSVArtifact`  ('Features of the dirty training dataset which we need to clean. Each dirty cell contains an embedded list of clean candidate values.',)
``X_train_clean``  :class:`dcbench.CSVArtifact`  Features of the clean training dataset where each dirty value from the dirty dataset is replaced with the correct clean candidate.
``y_train``        :class:`dcbench.CSVArtifact`  Labels of the training dataset.
``X_val``          :class:`dcbench.CSVArtifact`  Feature of the validtion dataset which can be used to guide the cleaning optimization process.
``y_val``          :class:`dcbench.CSVArtifact`  Labels of the validation dataset.
``X_test``         :class:`dcbench.CSVArtifact`  ('Features of the test dataset used to produce the final evaluation score of the model.',)
``y_test``         :class:`dcbench.CSVArtifact`  Labels of the test dataset.
=================  ============================  ========================================================================================================================================

Solution Artifacts
____________________
================  ============================  =============
name              type                          description
================  ============================  =============
``idx_selected``  :class:`dcbench.CSVArtifact`
================  ============================  =============

TODO: Provide more details on how to run budgetclean evaluation. 
