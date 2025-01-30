import numpy as np
import pickle
import os
from numpy.random import default_rng
from load_data import read_dataset
from classification import DecisionTreeClassifier
from evaluation import confusion_matrix, accuracy_from_confusion, accuracy, precision, recall, f1_score
from improvement import train_and_predict
from print_tree import print_tree
from split_dataset import split_dataset
from find_best_node import find_best_node
from load_train_models import load_train_models
from cross_validation import train_test_k_fold


if __name__ == "__main__":

    print("Loading the train_full dataset...")
    full_instances, full_class_labels_index_mapped, full_unique_classes, full_class_labels_str = read_dataset(
        f"data/train_full.txt")

    print("Loading the train_noisy dataset...")
    noisy_instances, noisy_class_labels_index_mapped, noisy_unique_classes, noisy_class_labels_str = read_dataset(
        f"data/train_noisy.txt")

    print("Loading the train_sub dataset...")
    sub_instances, sub_class_labels_index_mapped, sub_unique_classes, sub_class_labels_str = read_dataset(
        f"data/train_sub.txt")

    print("Loading the validation dataset...")
    val_instances, val_class_labels_index_mapped, val_unique_classes, val_class_labels_str = read_dataset(f"data/validation.txt")


    full = "full"
    noisy = "noisy"
    sub = "sub"

    full_tree = load_train_models(full, full_instances, full_class_labels_str)
    noisy_tree = load_train_models(noisy, noisy_instances, noisy_class_labels_str)
    sub_tree = load_train_models(sub, sub_instances, sub_class_labels_str)

    # then make predictions as the trees will all have same variable name
    print("Loading the test dataset...")
    test_instances, test_class_labels_index_mapped, unique_test_classes, test_class_labels_str \
        = read_dataset("data/test.txt")

    full_predictions = full_tree.predict(test_instances)
    noisy_predictions = noisy_tree.predict(test_instances)
    sub_predictions = sub_tree.predict(test_instances)

    name_list = [full, noisy, sub]
    predictions_list = [full_predictions, noisy_predictions, sub_predictions]

    for x in range(3):
        # now print out the confusion matrix, accuracy, precision, recall, f1, etc.
        _accuracy = accuracy(test_class_labels_str, predictions_list[x])
        print(f"Accuracy of {name_list[x]} dataset predictions: {_accuracy}")
        _confusion = confusion_matrix(test_class_labels_str, predictions_list[x])
        print(f"Confusion matrix of {name_list[x]} dataset predictions: \n"
          f"{_confusion}")
        _accuracy_from_confusion = accuracy_from_confusion(_confusion)
        print(f"{name_list[x]} accuracy from confusion matrix: {_accuracy_from_confusion}")
        _precision, _macro_precision = precision(test_class_labels_str, predictions_list[x])
        print(f"Precision of {name_list[x]} dataset predictions: {_precision}")
        print(f"Macro precision of {name_list[x]} dataset predictions: {_macro_precision}")
        _recall, _macro_recall = recall(test_class_labels_str, predictions_list[x])
        print(f"Recall of {name_list[x]} dataset predictions: {_recall}")
        print(f"Macro recall of {name_list[x]} dataset predictions: {_macro_recall}")
        _f1, _macro_f1 = f1_score(test_class_labels_str, predictions_list[x])
        print(f"F1 score of {name_list[x]} dataset predictions: {_f1}")
        print(f"Macro f1 of {name_list[x]} dataset predictions: {_macro_f1}")

    seed = 101
    rg = default_rng(seed)

    n_folds = 10
    accuracies = np.zeros((n_folds,))
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(full_instances))):
        # get the dataset from the correct splits
        x_train = full_instances[train_indices, :]
        y_train = full_class_labels_str[train_indices]
        x_test = full_instances[test_indices, :]
        y_test = full_class_labels_str[test_indices]

        # Train the decision tree model
        model_name = f"cross_validation_full_k={i}"
        model_tree = load_train_models(model_name, x_train, y_train)
        predictions = model_tree.predict(x_test)
        acc = accuracy(y_test, predictions)
        accuracies[i] = acc

    print(accuracies)
    print(accuracies.mean())
    print(accuracies.std())

    print("Combining the predictions on test.txt for all 10 decision trees trained in cross-validation")
    cross_validation_predictions_list = []
    for i in range(10):
        model_name = f"cross_validation_full_k={i}"
        model_tree = load_train_models(model_name)
        predictions = model_tree.predict(test_instances)
        cross_validation_predictions_list.append(predictions)

    #combine predictions into a 2d numpy array
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
    _accuracy = accuracy(test_class_labels_str, mode_predictions)
    print(f"Accuracy of combined cross validation predictions: {_accuracy}")
    _confusion = confusion_matrix(test_class_labels_str, mode_predictions)
    print(f"Confusion matrix of combined cross validation predictions: \n"
          f"{_confusion}")
    _accuracy_from_confusion = accuracy_from_confusion(_confusion)
    print(f"Combined cross validation accuracy from confusion matrix: {_accuracy_from_confusion}")
    _precision, _macro_precision = precision(test_class_labels_str, mode_predictions)
    print(f"Precision of combined cross validation predictions: {_precision}")
    print(f"Macro precision of combined cross validation predictions: {_macro_precision}")
    _recall, _macro_recall = recall(test_class_labels_str, mode_predictions)
    print(f"Recall of combined cross validation predictions: {_recall}")
    print(f"Macro recall of combined cross validation predictions: {_macro_recall}")
    _f1, _macro_f1 = f1_score(test_class_labels_str, mode_predictions)
    print(f"F1 score of combined cross validation predictions: {_f1}")
    print(f"Macro f1 of combined cross validation predictions: {_macro_f1}")



    full_improved_predictions_pruning = train_and_predict(full_instances, full_class_labels_str, test_instances, val_instances, val_class_labels_str, full)
    _accuracy = accuracy(test_class_labels_str, full_improved_predictions_pruning)
    print(f"Accuracy of pruned improved tree on full set: {_accuracy}")

    noisy_improved_predictions_pruning = train_and_predict(noisy_instances, noisy_class_labels_str, test_instances, val_instances, val_class_labels_str, noisy)
    _accuracy = accuracy(test_class_labels_str, noisy_improved_predictions_pruning)
    print(f"Accuracy of pruned improved tree on noisy set: {_accuracy}")

    sub_improved_predictions_pruning = train_and_predict(sub_instances, sub_class_labels_str, test_instances,
                                                           val_instances, val_class_labels_str, sub)
    _accuracy = accuracy(test_class_labels_str, sub_improved_predictions_pruning)
    print(f"Accuracy of pruned improved tree on sub set: {_accuracy}")


















