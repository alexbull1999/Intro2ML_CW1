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

if __name__ == "__main__":
    print("Loading the training dataset...")
    training_instances, class_labels_index_mapped, unique_classes, class_labels_str = read_dataset("data/train_full.txt")


    if os.path.exists("models/train_full_tree.pkl"):
        print("Found existing decision tree model for train_full data, loading from pickle file:")
        with open("models/train_full_tree.pkl", "rb") as file:
            train_full_tree = pickle.load(file)
        #print("Training decision tree model...")
        #print_tree(train_full_tree)
    else:
        print("Training the decision tree with max depth...")
        classifier = DecisionTreeClassifier()
        train_full_tree = classifier.fit(training_instances, class_labels_str)
        print("Saving the decision tree...")
        with open("models/train_full_tree.pkl", "wb") as file:
            pickle.dump(train_full_tree, file)
        #print("Printing tree")
        #print_tree(train_full_tree)


    #repeat for train noisy, train sub_trees


    # then make predictions as the trees will all have same variable name
    print("Loading the test dataset...")
    test_instances, test_class_labels_index_mapped, unique_test_classes, test_class_labels_str \
        = read_dataset("data/test.txt")
    predictions = train_full_tree.predict(test_instances)

    # now print out the confusion matrix, accuracy, precision, recall, f1, etc.
    train_full_accuracy = accuracy(test_class_labels_str, predictions)
    print(f"Accuracy of train full dataset predictions: {train_full_accuracy}")
    train_full_confusion = confusion_matrix(test_class_labels_str, predictions)
    print(f"Confusion matrix of train full dataset predictions: \n"
          f"{train_full_confusion}")
    train_full_accuracy_from_confusion = accuracy_from_confusion(train_full_confusion)
    print(f"Train full accuracy from confusion matrix: {train_full_accuracy_from_confusion}")
    train_full_precision, train_full_macro_precision = precision(test_class_labels_str, predictions)
    print(f"Precision of train full dataset predictions: {train_full_precision}")
    print(f"Macro precision of train full dataset predictions: {train_full_macro_precision}")
    train_full_recall, train_full_macro_recall = recall(test_class_labels_str, predictions)
    print(f"Recall of train full dataset predictions: {train_full_recall}")
    print(f"Macro recall of train full dataset predictions: {train_full_macro_recall}")
    train_full_f1, train_full_macro_f1 = f1_score(test_class_labels_str, predictions)
    print(f"F1 score of train full dataset predictions: {train_full_f1}")
    print(f"Macro f1 of train full dataset predictions: {train_full_macro_f1}")
