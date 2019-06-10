from joblib import dump, load


class SklearnWrapper(object):
    """
    A convenience wrapper class for various Scikit-learn's models (RandomForest, Logistic Regression, etc.)

    Parameters
    ----------
    clf: A scikit-learn classifier
    seed: int
        random seed
    params: dict
        parameters for the classifier, please read scikit-learn's documentation for more information
    """
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, **kwargs):
        """
        Trains the model

        Parameters
        ----------
        x_train numpy.array
            input parameters
        y_train numpy.array
            training labels

        Returns
        -------

        """
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        """
        Parameters
        ----------
        x numpy.array
            input parameters / features

        Returns
        -------
        y numpy.array
            predicted values
        """
        return self.clf.predict(x)

    def save(self, filepath):
        """
        Saves the model's parameters to hard drive for later use

        Parameters
        ----------
        filepath str
            path + filename, e.g.: "models/saved_models/random_forest.pkl"

        Returns
        -------

        """
        dump(self.clf, filepath)  # path + filename

    def load(self, filepath):
        """
        Load the model's parameters from hard drive

        Parameters
        ----------
        filepath str
            path + filename, e.g.: "models/saved_models/random_forest.pkl"
        Returns
        -------

        """
        self.clf = load(filepath)
