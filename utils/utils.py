import time
import json
import numpy as np
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score


@contextmanager
def timer(task_name="timer"):
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


def json_to_dict(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)


def find_besh_threshold_naive(labels, predictions):
    max_auc = 0
    max_thres = 0.1
    for threshold in np.linspace(0.0, 0.99, 100):
        binarized_oof = (predictions >= threshold).astype(int)
        if roc_auc_score(labels, binarized_oof) > max_auc:
            max_auc = roc_auc_score(labels, binarized_oof)
            max_thres = threshold
    return max_thres