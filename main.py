import numpy as np

from load_data import read_dataset
from classification import DecisionTreeClassifier
from improvement import train_and_predict

if __name__ == "__main__":
    print("Loading the training dataset...")
    training_instances, class_labels, unique_classes = read_dataset("data/train_full.txt")

    print(training_instances.shape)
    print(class_labels.shape)
    print(unique_classes.shape)

