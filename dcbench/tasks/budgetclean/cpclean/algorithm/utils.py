from scipy.stats import entropy
from collections import Counter
import numpy as np

def compute_entropy_by_counts(counts):
    """Compute entropy given counts of each label

    Args:
        counts (dict): {label: count}
    """
    s = sum(counts.values())
    if s == 0:
        return float("inf")
    p = [c/s for c in counts.values()]
    return entropy(p)

def compute_entropy_by_labels(A):
    """ Compute entropy over a list of labels

    Args:
        A (list): a list of labels (e.g. [0, 0, 1, 1, 0])
    """
    c = Counter(A)
    p = [x/len(A) for x in c.values()]
    return entropy(p)

def product(a):
    """Compute the product of all element in an integer array
    
    Args:
        A (list): a list of integers
    """
    if 0 in a:
        return 0

    count = Counter(a)
    result = 1
    for k, v in count.items():
        result *= k**v
    return result

def majority_vote(A):
    """Take the majority vote from a list of labels

    Args:
        A (list): a list of labels (e.g. [0, 0, 1, 1, 0])
    """
    major = np.argmax(np.bincount(A))
    return int(major)



