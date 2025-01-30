import numpy as np
import os
from load_data import read_dataset
import pickle
from improved_classification_q4 import *
from print_tree import print_tree



def load_train_improved_models(model_name, instances=None, class_labels_str=None, val_instances=None, val_class_labels_str=None):

    if os.path.exists(f"models/train_{model_name}_tree.pkl"):
        #print(f"Found existing decision tree model for {model_name} data, loading from pickle file:")
        with open(f"models/train_{model_name}_tree.pkl", "rb") as file:
            tree = pickle.load(file)
        #print("Training decision tree model...")
        #print_tree(tree)
        return tree
    elif instances is not None and class_labels_str is not None and val_instances is not None and val_class_labels_str is not None:
        print(f"Training the {model_name} decision tree...")
        classifier = ImprovedDecisionTreeClassifier()
        tree = classifier.improved_fit(instances, class_labels_str)
        tree.prune(instances, class_labels_str)
        #print("Saving the decision tree...")
        with open(f"models/train_{model_name}_tree.pkl", "wb") as file:
            pickle.dump(tree, file)
        # print("Printing tree")
        # print_tree(train_full_tree)
        return tree
    else:
        print(f"Error: No model found of name {model_name} and not enough parameters passed in to train and prune a new model")
        return None