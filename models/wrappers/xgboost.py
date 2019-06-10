import xgboost as xgb
from joblib import dump, load


class XGBWrapper(object):
    """
    A convenience wrapper class for XGBoost

    Parameters
    ----------
    params: dict
        parameters for the classifier, please read XGB's documentation for more information
    """
    def __init__(self, params):
        self.nrounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)
        self.param = params
        self.model = None

    def train(self, x_train, y_train, **kwargs):
        """
        Trains the model

        Parameters
        ----------
        x_train numpy.array
            input parameters
        y_train numpy.array
            training labels
        kwargs dict
            an optional dict which may contain 'x_val' (validation input features) 'y_val' (validation labels)
            if they are provided in the dict, XGBoost will run with validation dataset for early stopping mechanism
        Returns
        -------

        """
        dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist = None

        if 'x_val' in kwargs and 'y_val' in kwargs:
            dvalid = xgb.DMatrix(data=kwargs['x_val'], label=kwargs['y_val'])
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        if watchlist is None:
            print('training without validation dataset')
            self.model = xgb.train(dtrain=dtrain, num_boost_round=self.nrounds,
                                   verbose_eval=1000, params=self.param)
        else:
            print('training with validation dataset')
            self.model = xgb.train(dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist,
                                   early_stopping_rounds=self.early_stop_rounds,
                                   verbose_eval=1000, params=self.param)

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
        return self.model.predict(xgb.DMatrix(x), ntree_limit=self.model.best_ntree_limit)

    def save(self, filepath):
        """
        Saves the model's parameters to hard drive for later use

        Parameters
        ----------
        filepath str
            path + filename, e.g.: "models/saved_models/xgb.pkl"

        Returns
        -------

        """
        if self.model is None:
            print('model has never been trained before, returning...')
            return
        dump(self.model, filepath)  # path + filename

    def load(self, filepath):
        """
        Load the model's parameters from hard drive

        Parameters
        ----------
        filepath str
            path + filename, e.g.: "models/saved_models/xgb.pkl"
        Returns
        -------

        """
        self.model = load(filepath)