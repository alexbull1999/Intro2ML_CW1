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
        print("Training decision tree model...")
        print_tree(train_full_tree)

    else:
        print("Training the decision tree...")
        classifier = DecisionTreeClassifier()
        train_full_tree = classifier.fit(training_instances, class_labels_str)
        print("Saving the decision tree...")
        with open("models/train_full_tree.pkl", "wb") as file:
            pickle.dump(train_full_tree, file)
        #print("Printing tree")
        #print_tree(train_full_tree)

    #repeat for train noisy, train sub

    #then make predictions as the trees will all have same variable name






    """
       feature_index, threshold = find_best_node(training_instances, class_labels_str)
       print(f"Feature index: {feature_index}")
       print(f"Threshold: {threshold}")
       left_dataset, left_labels, right_dataset, right_labels = split_dataset(training_instances, class_labels_str, feature_index, threshold)
       print(f"Left dataset: {left_dataset}")
       print(f"Left labels: {left_labels}")
       print(f"Len unique left labels: {len(np.unique(left_labels))}")
       print(f"Right dataset: {right_dataset}")
       print(f"Right labels: {right_labels}")

       feature_index, threshold = find_best_node(right_dataset, right_labels)
       print(f"Feature index: {feature_index}")
       print(f"Threshold: {threshold}")
       new_left_dataset, new_left_labels, new_right_dataset, new_right_labels = split_dataset(right_dataset, right_labels, feature_index, threshold)
       print(f"New left dataset: {new_left_dataset}")
       print(f"New left labels: {new_left_labels}")
       print(f"New right dataset: {new_right_dataset}")
       print(f"New right labels: {new_right_labels}")


       feature_index, threshold = find_best_node(new_left_dataset, new_left_labels)
       print(f"Feature index: {feature_index}")
       print(f"Threshold: {threshold}")
       new_left_dataset, new_left_labels, new_right_dataset, new_right_labels = split_dataset(new_left_dataset, new_left_labels, feature_index, threshold)
       print(f"New left dataset: {new_left_dataset}")
       print(f"New left labels: {new_left_labels}")
       print(f"New right dataset: {new_right_dataset}")
       print(f"New right labels: {new_right_labels}")
       """




