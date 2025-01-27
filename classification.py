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

    def __init__(self, depth=0):
        self.is_trained = False
        self.feature = None
        self.threshold = None
        self.label = None
        self.depth = depth
        self.children = []

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

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        # 2 base cases - all samples have same label or dataset cannot be split further
        # base case 1: all labels are same
        if len(np.unique(y)) == 1:
            self.label=y[0] #in this case all labels identical in the sample, so can return any
            #print(f"Depth = {self.depth}")
            return

        # base case 2: labels are different but all features are identical
        elif np.all(x == x[0, :], axis=0).all():  # Check if all rows are identical across all columns
            # Return majority class label
            unique, counts = np.unique(y, return_counts=True)
            self.label=unique[np.argmax(counts)]
            return


        #case 3: we have reached our max_depth of 5 and want to return the majority label
        # (max_depth crudely implemented for part 2 of the CW! Not for our part 4 improvements)
        # we did 17 as we saw it had just as good a performance as a 21 max-depth (that was the depth of original tree)
        #but reduces overfitting to a minor extent
        elif self.depth == 17:
            # Return majority class label
            unique, counts = np.unique(y, return_counts=True)
            self.label = unique[np.argmax(counts)]
            return


        #else we attempt to find the best node / split point that maximises the information gain and record what it is
        else:
            feature_index, threshold = find_best_node(x, y)
            if feature_index is None and threshold is None:
                #if no split point can improve IG return majority class label
                unique, counts = np.unique(y, return_counts=True)
                self.label=unique[np.argmax(counts)]
                return

            else:
                self.feature=feature_index
                self.threshold=threshold
                # then we need to split the dataset based on this split point,
                left_dataset, left_labels, right_dataset, right_labels = split_dataset(x, y, feature_index, threshold)

                # recursively create child nodes
                left_child = DecisionTreeClassifier(self.depth+1)
                left_child.fit(left_dataset, left_labels)

                right_child = DecisionTreeClassifier(self.depth+1)
                right_child.fit(right_dataset, right_labels)

                self.children = [left_child, right_child]

        return self

    
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

        for row in range(len(x)):
            tree_position = self
            while tree_position.is_leaf() is False:
                if x[row, tree_position.feature] < tree_position.threshold:
                    tree_position = tree_position.children[0]
                else:
                    tree_position = tree_position.children[1]
            predictions[row] = tree_position.label
    
        # remember to change this if you rename the variable
        return predictions
        


