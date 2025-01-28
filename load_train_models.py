import numpy as np
import os
from load_data import read_dataset
import pickle
from classification import *


def load_train_models(model_name):
    instances, class_labels_index_mapped, unique_classes, class_labels_str = read_dataset(
        f"data/train_{model_name}.txt")

    if os.path.exists(f"models/train_{model_name}_tree.pkl"):
        print(f"Found existing decision tree model for {model_name} data, loading from pickle file:")
        with open(f"models/train_{model_name}_tree.pkl", "rb") as file:
            tree = pickle.load(file)
        # print("Training decision tree model...")
        # print_tree(train_full_tree)
    else:
        print(f"Training the {model_name} decision tree with max depth...")
        classifier = DecisionTreeClassifier()
        tree = classifier.fit(instances, class_labels_str)
        print("Saving the decision tree...")
        with open(f"models/train_{model_name}_tree.pkl", "wb") as file:
            pickle.dump(f"train_{model_name}", file)
        # print("Printing tree")
        # print_tree(train_full_tree)

    return tree