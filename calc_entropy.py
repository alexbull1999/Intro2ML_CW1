import numpy as np

def calculate_entropy(dataset):

    indices_map, freq = np.unique(dataset, return_counts=True)
    #print(dataset, dataset.shape)
    #print(indices_map, freq)
    entropy = 0
    for i in range(len(indices_map)):
        change = ((freq[i] / len(dataset)) * np.log2(freq[i] / len(dataset)))
        entropy = entropy - change

    return entropy