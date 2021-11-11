🎯 Tasks
=========

Minimal Cleaning
################
Summary: 
        When it comes to data preparation, data cleaning is often an essential yet 
        quite costly task. If we are given a fixed cleaning budget, the challenge is to 
        find the training data examples that would would bring the biggest positive 
        impact on model performance if we were to clean them.
    

:Number of problems: 8

Artefacts
______________
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

Attributes
______________



Detailed description 


Slice Discovery
################
Summary: Machine learnings models that achieve high overall accuracy often make  systematic erors on important subgroups (or *slices*) of data. When working with  high-dimensional inputs (*e.g.* images, audio) where data slices are often  unlabeled, identifying underperforming slices is a challenging. In this task,  we'll develop automated slice discovery methods that mine unstructured data for  underperforming.

:Number of problems: 4

Artefacts
______________
================  ======================================  =======================================================================
name              type                                    description
================  ======================================  =======================================================================
``predictions``   :class:`dcbench.DataPanelArtefact`      A Datapanel of the model's predictions with columns `"id"`,`"target"`,
                                                                          and `"probs".`
``slices``        :class:`dcbench.DataPanelArtefact`      A datapanel containing ground truth slice labels
``activations``   :class:`dcbench.DataPanelArtefact`
``base_dataset``  :class:`dcbench.VisionDatasetArtefact`  A base dataset
================  ======================================  =======================================================================

Attributes
______________



Detailed description 


Minimal Dataset Selection
################
Summary: Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance.

:Number of problems: 1

Artefacts
______________
==============  ==================================  ==============
name            type                                description
==============  ==================================  ==============
``train_data``  :class:`dcbench.DataPanelArtefact`  Training data.
``test_data``   :class:`dcbench.DataPanelArtefact`  Testing data.
==============  ==================================  ==============

Attributes
______________



Detailed description 
