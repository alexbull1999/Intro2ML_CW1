import numpy as np

from load_data import read_dataset
from classification import DecisionTreeClassifier
from improvement import train_and_predict

if __name__ == "__main__":
    print("Loading the training dataset...")
    training_instances, class_labels_index_mapped, unique_classes, class_labels_str = read_dataset("data/train_full.txt")

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(training_instances, class_labels_str)



