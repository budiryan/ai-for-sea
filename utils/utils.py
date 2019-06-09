import time
import json
from contextlib import contextmanager


@contextmanager
def timer(task_name="timer"):
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


def json_to_dict(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)