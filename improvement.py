##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from improved_classification_q4 import ImprovedDecisionTreeClassifier
from numpy.random import default_rng
from cross_validation import train_test_k_fold
from load_train_improved_models_q4 import load_train_improved_models
from evaluation import accuracy


def train_and_predict(x_train, y_train, x_test, x_val, y_val, name=None):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
       

    # TODO: Train new classifier

    #Combine validation and train into one dataset, to maximise size, and then perform cross validation
    combined_train_val_instances = np.concatenate((x_train, x_val))
    combined_train_val_class_labels = np.concatenate((y_train, y_val))

    #TODO:Iterate carousel/cross-validation style

    seed = 101
    rg = default_rng(seed)

    n_folds = 10
    accuracies = np.zeros((n_folds,))
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(combined_train_val_instances))):
        # get the dataset from the correct splits
        final_x_train = combined_train_val_instances[train_indices, :]
        final_y_train = combined_train_val_class_labels[train_indices]
        final_x_val = combined_train_val_instances[test_indices, :]
        final_y_val = combined_train_val_class_labels[test_indices]

        # Train the decision tree model
        if name is None:
            model_name = f"pruned_cross_validation_k={i}"
        else:
            model_name = f"pruned_{name}_cross_validation_k={i}"
        #Try pruning on the full train dataset rather than the validation? TEST on val?
        classifier = ImprovedDecisionTreeClassifier()
        model_tree = load_train_improved_models(model_name, final_x_train, final_y_train, final_x_val, final_y_val)
        # note pruning takes place in the load_train_improved_models script
        predictions = model_tree.improved_predict(final_x_val)
        acc = accuracy(final_y_val, predictions)
        accuracies[i] = acc

    print(accuracies)
    print(accuracies.mean())
    print(accuracies.std())

    #these accuracies are based on cross-validation only
    # We can now use a random forest approach to predict labels on x_test

    print("Combining the predictions on test.txt for all 10 pruned decision trees trained")
    cross_validation_predictions_list = []
    for i in range(10):
        if name is None:
            model_name = f"pruned_cross_validation_k={i}"
        else:
            model_name = f"pruned_{name}_cross_validation_k={i}"
        model_tree = load_train_improved_models(model_name)
        predictions = model_tree.improved_predict(x_test)
        cross_validation_predictions_list.append(predictions)

    # combine predictions into a 2d numpy array
    predictions_array = np.stack(cross_validation_predictions_list, axis=1)
    mode_predictions = []
    # Iterate over each row (instance) in the predictions array
    for row in predictions_array:
        # Use np.unique to find unique values and their counts
        unique_values, counts = np.unique(row, return_counts=True)
        # Find the mode (value with the highest count)
        mode_value = unique_values[np.argmax(counts)]
        mode_predictions.append(mode_value)

    mode_predictions = np.array(mode_predictions)


    # set up an empty (M, ) numpy array to store the predicted labels
    # # feel free to change this if needed
    #   predictions = np.zeros((x_test.shape[0],), dtype=object)

    # TODO: Make predictions on x_test using new classifier

    # remember to change this if you rename the variable
    return mode_predictions


