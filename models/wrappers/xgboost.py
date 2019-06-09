import xgboost as xgb
from joblib import dump, load


class XGBWrapper(object):
    def __init__(self, params):
        self.nrounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)
        self.param = params
        self.model = None

    def train(self, x_train, y_train, **kwargs):
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
        return self.model.predict(xgb.DMatrix(x), ntree_limit=self.model.best_ntree_limit)

    def save(self, filepath):
        if self.model is None:
            print('model has never been trained before, returning...')
            return
        dump(self.model, filepath)  # path + filename

    def load(self, filepath):
        self.model = load(filepath)