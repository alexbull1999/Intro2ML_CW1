import numpy as np

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - training_instances is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - class_labels is a numpy array with shape (N, ), and each element should be
                   an integer from 0 to C-1 where C is the number of classes
               - unique_classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """

    # TODO: Complete this function
    with open(filepath, 'r') as file:
      lines = file.readlines()

    #Parse dataset
    features = [] #empty array
    labels = [] #empty array
    for line in lines:
      line = line.strip() #Preprocessing to remove extra whitespace, incl. newlines
      if not line:
        continue
      values = line.split(',')
      features.append(tuple(map(int, values[:-1]))) #Extract features and convert to float (ignores final value in slice)
      labels.append(values[-1]) #extract the lables

    #Convert features to a NumPy array of tuples
    training_instances = np.array(features)

    #Map string labels to integer classification
    unique_classes, class_labels = np.unique(labels, return_inverse=True)

    return training_instances, class_labels, unique_classes