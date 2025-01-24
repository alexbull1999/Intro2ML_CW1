#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from find_best_node import find_best_node
from split_dataset import split_dataset
from calc_entropy import calculate_entropy

class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    feature: Feature index for splitting
    threshold = Threshold value for the split
    label = Label for leaf nodes (None for internal nodes)
    children = Child nodes (list of TreeNodes)
    
    Methods:
    is_leaf(self): returns true if this node is a leaf node (i.e. if self.label is not None)
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree

    """

    def __init__(self, feature=None, threshold=None, label=None, children=None):
        self.is_trained = False
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.children = children or []

    def is_leaf(self):
        return self.label is not None
    

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # 2 base cases - all samples have same label or dataset cannot be split further
        # base case 1: all labels are same
        if len(np.unique(y)) == 1:
            return DecisionTreeClassifier(label=y[0]) #in this case all labels identical in the sample, so can return any

        # base case 2: labels are different but all features are identical
        elif np.all(x == x[0, :], axis=0).all():  # Check if all rows are identical across all columns
            # Return majority class label
            unique, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier(label=unique[np.argmax(counts)])

        #else we attempt to find the best node / split point that maximises the information gain and record what it is
        else:
            feature_index, threshold = find_best_node(x, y)
            if feature_index is None and threshold is None:
                #if no split point can improve IG return majority class label
                unique, counts = np.unique(y, return_counts=True)
                return DecisionTreeClassifier(label=unique[np.argmax(counts)])

            else:
                node = DecisionTreeClassifier(feature=feature_index, threshold=threshold)
                # then we need to split the dataset based on this split point,
                left_dataset, left_labels, right_dataset, right_labels = split_dataset(x, y, feature_index, threshold)

                # finally we can recursively create child nodes
                node.children = [self.fit(left_dataset, left_labels), self.fit(right_dataset, right_labels)]



        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return node

    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################




    
        # remember to change this if you rename the variable
        return predictions
        


