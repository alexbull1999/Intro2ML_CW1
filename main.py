import numpy as np
import pickle
import os
from load_data import read_dataset
from classification import DecisionTreeClassifier
from evaluation import confusion_matrix, accuracy_from_confusion, accuracy, precision, recall, f1_score
from improvement import train_and_predict
from print_tree import print_tree
from split_dataset import split_dataset
from find_best_node import find_best_node
from load_train_models import load_train_models

if __name__ == "__main__":
    full = "full"
    noisy = "noisy"
    sub = "sub"

    full_tree = load_train_models(full)
    noisy_tree = load_train_models(noisy)
    sub_tree = load_train_models(sub)

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



