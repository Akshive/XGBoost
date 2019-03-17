# XGBoost
Basics of XGBoost



List of Params

silent = 0, 1   1--->silent mode(no o/p)
eta = 0,..0.3,....1       step shrinkage, used in update to prevent overfitting.  smaller values ---> slower learning.

max_depth = 1,...6,.....inf               max_depth of undelying decision trees
min_child_weight = 0,...1....inf          min sum of weights needed for partitioning.       higher value ---> slower learning.

lambda = L2 regularization param
alpha = L1 regularization param
