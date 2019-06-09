import numpy as np
from sklearn.model_selection import StratifiedKFold


def train_cv_predict_test(clf, num_splits, X, y, X_test):
    oof_train = np.zeros((len(X),))
    oof_test = np.zeros((len(X_test),))
    oof_test_skf = np.empty((num_splits, len(X_test)))

    for i, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=num_splits, shuffle=True).split(X, y)):
        print('Training for fold: ', i + 1)

        x_tr = X[train_index, :]
        y_tr = y[train_index]
        x_te = X[test_index, :]
        y_te = y[test_index]

        clf.train(x_tr, y_tr, x_val=x_te, y_val=y_te)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def train_cv(clf, num_splits, X, y):
    oof_train = np.zeros((len(X),))

    for i, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=num_splits, shuffle=True).split(X, y)):
        print('Training for fold: ', i + 1)

        x_tr = X[train_index, :]
        y_tr = y[train_index]
        x_te = X[test_index, :]
        y_te = y[test_index]

        clf.train(x_tr, y_tr, x_val=x_te, y_val=y_te)

        oof_train[test_index] = clf.predict(x_te)

    return oof_train.reshape(-1, 1)