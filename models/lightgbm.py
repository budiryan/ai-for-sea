import lightgbm as lgb
from joblib import dump, load


class LGBWrapper(object):
    def __init__(self, params):
        self.num_rounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)
        self.param = params
        self.model = None

    def train(self, x_train, y_train, **kwargs):
        dtrain = lgb.Dataset(x_train, label=y_train)
        watchlist = None

        if 'x_val' in kwargs and 'y_val' in kwargs:
            dvalid = lgb.Dataset(kwargs['x_val'], label=kwargs['y_val'])
            watchlist = [dtrain, dvalid]

        if watchlist is None:
            print('training without validation dataset...')
            self.model = lgb.train(
                      self.param,
                      train_set=dtrain,
                      num_boost_round=self.num_rounds,
                      verbose_eval=1000,
            )
        else:
            print('training with validation dataset...')
            print('self param: ', self.param)
            self.model = lgb.train(
                      self.param,
                      train_set=dtrain,
                      num_boost_round=self.num_rounds,
                      early_stopping_rounds=self.early_stop_rounds,
                      valid_sets=watchlist,
                      verbose_eval=1000,
            )

    def predict(self, x):
        if self.model is None:
            print('model has never been trained before, returning...')
            return
        return self.model.predict(x, num_iteration=self.model.best_iteration)

    def save(self, filepath):
        dump(self.model, filepath)  # path + filename

    def load(self, filepath):
        self.model = load(filepath)