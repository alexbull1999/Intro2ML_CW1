import numpy as np

from load_data import read_dataset
from classification import DecisionTreeClassifier
from improvement import train_and_predict

def compare_and_visualise_data():
    print("Loading the training dataset...")
    training_data_instances, training_class_labels, training_unique_classes = read_dataset("data/train_full.txt")
    sub_instances, sub_class_labels, sub_unique_classes = read_dataset("data/train_sub.txt")
    noisy_instances, noisy_class_labels, noisy_unique_classes = read_dataset("data/train_noisy.txt")

    print("Number of instances per dataset:")
    print(f"Training data has: {len(training_data_instances)} instances")
    print(f"Sub data has: {len(sub_instances)} instances")
    print(f"Noisy data has: {len(noisy_instances)} instances")

    print("Number of unique class labels per dataset:")
    print(f"Training data has: {len(training_unique_classes)} unique class labels, that are {training_unique_classes}")
    print(f"Sub data has: {len(sub_unique_classes)} unique class labels, that are {sub_unique_classes}")
    print(f"Noisy data has: {len(noisy_unique_classes)} unique class labels, that are {noisy_unique_classes}")

    print("Distribution of instances across classes per dataset:")

    indices_map, freq = np.unique(training_class_labels, return_counts=True)
    training_percentage_split = np.round((freq/len(training_class_labels)) * 100, 3)
    labels = ['A', 'C', 'E', 'G', 'O', 'Q']
    for i in range(6):
        print(f"Training data has: {training_percentage_split[i]}% instances of {labels[i]}")

    sub_indices_map, sub_freq = np.unique(sub_class_labels, return_counts=True)
    sub_percentage_split = np.round((sub_freq / len(sub_class_labels)) * 100, 3)
    for i in range(6):
        print(f"Sub data has: {sub_percentage_split[i]}% instances of {labels[i]}")

    noisy_indices_map, noisy_freq = np.unique(noisy_class_labels, return_counts=True)
    noisy_percentage_split = np.round((noisy_freq / len(noisy_class_labels)) * 100, 3)
    for i in range(6):
        print(f"Noisy data has: {noisy_percentage_split[i]}% instances of {labels[i]}")

    print("Range for each feature")
    difference = training_data_instances.max(axis=0) - training_data_instances.min(axis=0)
    formatted_difference = [f"{x:.2g}" for x in difference]
    print(formatted_difference)
    print(f"Max value per attribute: {training_data_instances.max(axis=0)}")
    print(f"Min value per attribute: {training_data_instances.min(axis=0)}")

    print(training_data_instances.dtype)

    percentage_difference = abs(training_percentage_split - noisy_percentage_split)
    class_change = (training_percentage_split - noisy_percentage_split)
    print(f"Percentage difference per class: {percentage_difference}")
    print(f"Class movements: {class_change}")



if __name__ == "__main__":
    compare_and_visualise_data()
