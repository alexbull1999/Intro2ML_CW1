import numpy as np
from calc_entropy import calculate_entropy


def find_best_node(data, labels):
    """ Calculate the information entropy of a dataset passed into the function

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                        N is the number of instances
                        K is the number of attributes
        labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                               Each element in y is a str

        Returns:
            feature index: integer representing feature/attribute index on which we are splitting the data at this node (e.g. column no. 13)
            threshold: integer the value at which we are splitting the data at this node; always splitting at (a < threshold)
    """

    # Calculate original entropy of dataset as a whole
    entropy = calculate_entropy(labels)

    # Sort by the first attribute (column 0)
    attribute_index = 0
    sorted_indices = np.argsort(data[:, attribute_index])

    # Sort x and y using the same indices
    data_sorted = data[sorted_indices]
    labels_sorted = labels[sorted_indices]


