import time
import json
import numpy as np
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score


@contextmanager
def timer(task_name="timer"):
    """
    A convenience function for displaying time spent executing a task

    Example:
        with timer("Execute a task"):
            do_task_1()
            do_task_2()
    """
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


def json_to_dict(filepath):
    """
    Parameters
    ----------
    filepath str

    Returns
    -------
    data dict
        a python dictionary containing data from the json file
    """
    with open(filepath) as json_file:
        return json.load(json_file)


def find_best_threshold_naive(labels, predictions):
    """ Finds best binary threshold for set of predictions given the labels in AUC metric
    Parameters
    ----------
    labels numpy.array
        A set of ground truth labels
    predictions
        A set of probability predictions

    Returns
    -------
    best_threshold int
        the best threshold for classifying 0 or 1
    """
    max_auc = 0
    best_threshold = 0.1
    for threshold in np.linspace(0.0, 0.99, 100):
        binarized_oof = (predictions >= threshold).astype(int)
        if roc_auc_score(labels, binarized_oof) > max_auc:
            max_auc = roc_auc_score(labels, binarized_oof)
            best_threshold = threshold
    return best_threshold
