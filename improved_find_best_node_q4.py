import numpy as np
from calc_entropy import calculate_entropy


def improved_find_best_node(data, labels):
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

    best_attribute_index = None
    best_threshold = None
    best_information_gain = 0

    # Sort by the first attribute (column 0)
    for attribute_index in range(data.shape[1]):
        sorted_indices = np.argsort(data[:, attribute_index])

        # Sort x and y using the same indices
        data_sorted = data[sorted_indices]
        labels_sorted = labels[sorted_indices]

        for row in range(len(labels_sorted)):
            #skip invalid splits that would leave an empty dataset
            if row == 0 or row == len(labels_sorted):
                continue


            #if there is a change in attribute value (i.e. a split point), calc entropy
            if data_sorted[row, attribute_index] != data_sorted[row-1, attribute_index]:
                #if labels_sorted[row] == labels_sorted[row-1]:
                    #continue ### Excluded this code following chat with Rob <-- see if extra logic required with Josiah
                # slice up to and not including row (i.e. where the change is)
                entropy_left = calculate_entropy(labels_sorted[0:row])
                entropy_right = calculate_entropy(labels_sorted[row:])
                weighted_entropy_left = (len(labels_sorted[0:row])/len(labels_sorted)) * entropy_left
                weighted_entropy_right = (len(labels_sorted[row:])/len(labels_sorted)) * entropy_right
                information_gain = entropy - (weighted_entropy_left + weighted_entropy_right)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_attribute_index = attribute_index
                    best_threshold = (data_sorted[row, attribute_index] + data_sorted[row-1, attribute_index])/2


    return best_attribute_index, best_threshold