import lightgbm as lgb


class LGBWrapper(object):
    def __init__(self, params=None):
        self.param = params
        self.num_rounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)

    def train(self, x_train, y_train, **kwargs):
        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(kwargs['x_val'], label=kwargs['y_val'])

        watchlist = [dtrain, dvalid]
        self.model = lgb.train(
                  self.param,
                  train_set=dtrain,
                  num_boost_round=self.num_rounds,
                  valid_sets=watchlist,
                  verbose_eval=1000,
                  early_stopping_rounds=self.early_stop_rounds
        )

    def predict(self, x):
        return self.model.predict(x, num_iteration=self.model.best_iteration)