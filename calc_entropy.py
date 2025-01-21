import numpy as np

from scipy.stats import entropy

def calculate_entropy(labels):
    """ Calculate the information entropy of a dataset passed into the function

        Args:
        labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                            N is the number of instances of data to calculate entropy of (equivalent to # of labels)
                           Each element in y is a str

        Returns:
            int: the information entropy of the dataset
    """

    num_instances = len(labels)
    A_count = C_count = E_count = G_count = O_count = Q_count = 0
    for label in labels:
        if (label == 'A'):
            A_count += 1
        elif (label == 'C'):
            C_count += 1
        elif (label == 'E'):
            E_count += 1
        elif (label == 'G'):
            G_count += 1
        elif (label == 'O'):
            O_count += 1
        elif (label == 'Q'):
            Q_count += 1
        else:
            print("Unknown label error")

    prob_a = A_count / num_instances
    prob_c = C_count / num_instances
    prob_e = E_count / num_instances
    prob_g = G_count / num_instances
    prob_o = O_count / num_instances
    prob_q = Q_count / num_instances

    """ (calculated by hand before I realised scipy has a method for doing this automatically!
    
    manual_entropy = ((-prob_a * np.log2(prob_a)) + (-prob_c * np.log2(prob_c)) + (-prob_e * np.log2(prob_e)) +
               (-prob_g * np.log2(prob_g)) + (-prob_o * np.log2(prob_o)) + (-prob_q * np.log2(prob_q)))
    """

    probabilities = [prob_a, prob_c, prob_e, prob_g, prob_o, prob_q]
    entropy_value = entropy(probabilities, base=2)

    #print(f"Manual entropy is: {manual_entropy}")
    #print(f"Scipy entropy is: {entropy_value}")

    return entropy_value







