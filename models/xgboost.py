import xgboost as xgb


class XGBWrapper(object):
    def __init__(self, params=None):
        self.param = params
        self.nrounds = params.pop('nrounds', 60000)
        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)

    def train(self, x_train, y_train, **kwargs):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(data=kwargs['x_val'], label=kwargs['y_val'])

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        self.model = xgb.train(dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist,
                               early_stopping_rounds=self.early_stop_rounds,
                               verbose_eval=1000, params=self.param)

    def predict(self, x):
        return self.model.predict(xgb.DMatrix(x), ntree_limit=self.model.best_ntree_limit)