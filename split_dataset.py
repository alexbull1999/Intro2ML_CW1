import numpy as np

def split_dataset(x, y, feature_index, threshold):
    """ Splits a instances (x) and labels (y) dataset in two based on the threshold value of the feature index provided
        Threshold value is always treated as a strict less than inequality (<)

            Args:
            x (numpy.ndarray): Instances, numpy array of shape (N, K) of the original dataset
                            N is the number of instances
                            K is the number of attributes
            labels (numpy.ndarray): Class labels, numpy array of shape (N, ) of the original dataset
                                   Each element in y is a str
            feature_index: An integer corresponding the column index of the feature on which the dataset is being split
            threshold: An integer corresponding to the value at which the dataset is being split (treated as <)

            Returns:
                left_dataset: The portion of x that is on the left hand side (less than) the threshold value
                left_labels: The corresponding portion of y
                right_dataset: The portion of x that is on the right hand side (greater than) the threshold value
                right_labels: The corresponding portion of y
    """

    #create a boolean mask based on whether values are above or below the threshold
    mask_left = x[:, feature_index] < threshold  # Boolean mask for the left dataset
    mask_right = ~mask_left  # Right: values strictly greater than threshold

    # print(mask_left)

    left_dataset = x[mask_left]
    right_dataset = x[mask_right]
    left_labels = y[mask_left]
    right_labels = y[mask_right]

    return left_dataset, left_labels, right_dataset, right_labels

