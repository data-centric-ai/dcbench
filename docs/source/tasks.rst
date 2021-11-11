🎯 Tasks
=========

Minimal Data Selection
--------------------------------------------
.. sidebar::
    Task Details
    
    :Task ID:      ``minidata``
    :Problems:     1
    :Cloud Storage: `browse <https://console.cloud.google.com/storage/browser/dcbench/minidata>`_

Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance.

**Classes**: :class:`dcbench.MiniDataProblem` :class:`dcbench.MiniDataSolution`




Problem Artefacts
__________________
==============  ==================================  =============================================================================
name            type                                description
==============  ==================================  =============================================================================
``train_data``  :class:`dcbench.DataPanelArtefact`  A DataPanel of train examples with columns ``id``, ``input``, and ``target``.
``test_data``   :class:`dcbench.DataPanelArtefact`  A DataPanel of test examples with columns ``id``, ``input``, and ``target``.
==============  ==================================  =============================================================================

Solution Artefacts
____________________
=============  =============================  ======================================================================
name           type                           description
=============  =============================  ======================================================================
``train_ids``  :class:`dcbench.YAMLArtefact`  A list of train example ids from the  ``id`` column of ``train_data``.
=============  =============================  ======================================================================

TODO: Provide more details on how to run miniddata evaluation.  

Slice Discovery
--------------------------------------------
.. sidebar::
    Task Details
    
    :Task ID:      ``slice_discovery``
    :Problems:     4
    :Cloud Storage: `browse <https://console.cloud.google.com/storage/browser/dcbench/slice_discovery>`_

Machine learnings models that achieve high overall accuracy often make  systematic erors on important subgroups (or *slices*) of data. When working   with high-dimensional inputs (*e.g.* images, audio) where data slices are   often unlabeled, identifying underperforming slices is a challenging. In  this task, we'll develop automated slice discovery methods that mine  unstructured data for underperforming slices.

**Classes**: :class:`dcbench.SliceDiscoveryProblem` :class:`dcbench.SliceDiscoverySolution`




Problem Artefacts
__________________
================  ======================================  =======================================================================================
name              type                                    description
================  ======================================  =======================================================================================
``predictions``   :class:`dcbench.DataPanelArtefact`      A DataPanel of the model's predictions with columns `id`,`target`, and `probs.`
``slices``        :class:`dcbench.DataPanelArtefact`      A DataPanel of the ground truth slice labels with columns  `id`, `target`, and `probs.`
``activations``   :class:`dcbench.DataPanelArtefact`      A DataPanel of the model's activations with columns `id`,`act`
``base_dataset``  :class:`dcbench.VisionDatasetArtefact`  A DataPanel representing the base dataset with columns `id` and `image`.
================  ======================================  =======================================================================================

Solution Artefacts
____________________
===============  ==================================  ==========================================================================
name             type                                description
===============  ==================================  ==========================================================================
``pred_slices``  :class:`dcbench.DataPanelArtefact`  A DataPanel of predicted slice labels with columns `id` and `pred_slices`.
===============  ==================================  ==========================================================================

TODO: Provide more details on how to run slice discovery evaluation. 

Minimal Feature Cleaning
--------------------------------------------
.. sidebar::
    Task Details
    
    :Task ID:      ``miniclean``
    :Problems:     8
    :Cloud Storage: `browse <https://console.cloud.google.com/storage/browser/dcbench/miniclean>`_

When it comes to data preparation, data cleaning is often an essential yet quite costly task. If we are given a fixed cleaning budget, the challenge is to find the training data examples that would would bring the biggest positive impact on model performance if we were to clean them.

**Classes**: :class:`dcbench.MinicleanProblem` :class:`dcbench.MiniCleanSolution`




Problem Artefacts
__________________
===================  ============================  =============
name                 type                          description
===================  ============================  =============
``X_train_dirty_a``  :class:`dcbench.CSVArtefact`
``X_train_dirty_b``  :class:`dcbench.CSVArtefact`
``X_train_clean_a``  :class:`dcbench.CSVArtefact`
``X_train_clean_b``  :class:`dcbench.CSVArtefact`
``y_train_a``        :class:`dcbench.CSVArtefact`
``y_train_b``        :class:`dcbench.CSVArtefact`
``X_val``            :class:`dcbench.CSVArtefact`
``y_val``            :class:`dcbench.CSVArtefact`
``X_test``           :class:`dcbench.CSVArtefact`
``y_test``           :class:`dcbench.CSVArtefact`
===================  ============================  =============

Solution Artefacts
____________________
=============  ============================  =============
name           type                          description
=============  ============================  =============
``train_ids``  :class:`dcbench.CSVArtefact`
=============  ============================  =============

TODO: Provide more details on how to run miniclean evaluation. 