import numpy as np
import pickle
import os
from load_data import read_dataset
from classification import DecisionTreeClassifier
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
        print("Training the decision tree...")
        classifier = DecisionTreeClassifier()
        train_full_tree = classifier.fit(training_instances, class_labels_str)
        print("Saving the decision tree...")
        with open("models/train_full_tree.pkl", "wb") as file:
            pickle.dump(train_full_tree, file)
        #print("Printing tree")
        #print_tree(train_full_tree)

    print(train_full_tree.is_trained)

    #repeat for train noisy, train sub

    #then make predictions as the trees will all have same variable name
    print("Loading the validation dataset...")
    validation_instances, validation_class_labels_index_mapped, unique_validation_classes, validation_class_labels_str \
        = read_dataset("data/validation.txt")
    predictions = train_full_tree.predict(validation_instances)
    print(predictions)
    print(validation_class_labels_str)
    print(np.sum(predictions == validation_class_labels_str))
    print(len(validation_class_labels_str))

    # then make predictions as the trees will all have same variable name
    print("Loading the test dataset...")
    test_instances, test_class_labels_index_mapped, unique_test_classes, test_class_labels_str \
        = read_dataset("data/test.txt")
    predictions = train_full_tree.predict(test_instances)
    print(predictions)
    print(test_class_labels_str)
    print(np.sum(predictions == test_class_labels_str))
    print(len(test_class_labels_str))
