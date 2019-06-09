from joblib import dump, load


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, **kwargs):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def save(self, filepath):
        dump(self.clf, filepath)  # path + filename

    def load(self, filepath):
        self.clf = load(filepath)
