import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
from utils.utils import timer, json_to_dict
from models.wrappers.lightgbm import LGBWrapper
from models.wrappers.xgboost import XGBWrapper
from models.wrappers.sklearn import SklearnWrapper
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from dataloader import load_features
from features.features import RideSafetyFeaturesAggregator


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='A script to predict test data')
    ap.add_argument('-s', '--source', required=True,
                    help='path to folder containing feature CSVs. Make sure the folder is STERILE and contains the test CSVs only')
    ap.add_argument("-d", "--destination", required=True,
                    help='path + filename for writing test prediction CSV. e.g.: "predictions.csv" ')
    args = vars(ap.parse_args())

    features_dir = args['source']
    write_dest = args['destination']

    if not os.path.isdir(features_dir):
        sys.exit('source is not a directory, existing...')

    with timer('Loading test data + generating features: '):
        features = load_features(features_dir)
        feature_aggregator = RideSafetyFeaturesAggregator(features)

        print('Generating features, it is going to take a while...')
        feat_aggs = feature_aggregator.get_aggregated_features()

    feature_columns = [c for c in feat_aggs.columns.values if c not in ['bookingID', 'label']]

    X = feat_aggs[feature_columns].values

    # predict XGB
    with timer('Predicting using XGB model'):
        params = json_to_dict('models/params/xgboost.json')
        model_files = glob.glob('{}/*.pkl'.format('models/saved_models/xgboost'))
        preds = []
        for f in model_files:
            clf = XGBWrapper(params)
            clf.load(f)
            pred = clf.predict(X)
            preds.append(pred)

        xgb_preds = np.average(preds, axis=0).reshape(-1, 1)

    # predict LGB
    with timer('Predicting using LGB model'):
        params = json_to_dict('models/params/lightgbm.json')
        model_files = glob.glob('{}/*.pkl'.format('models/saved_models/lightgbm'))
        preds = []
        for f in model_files:
            clf = LGBWrapper(params)
            clf.load(f)
            pred = clf.predict(X)
            preds.append(pred)

        lgb_preds = np.average(preds, axis=0).reshape(-1, 1)

    # predict Random forest
    with timer('Predicting using random forest model'):
        params = json_to_dict('models/params/rf.json')
        model_files = glob.glob('{}/*.pkl'.format('models/saved_models/rf'))
        preds = []
        for f in model_files:
            clf = SklearnWrapper(RandomForestRegressor, seed=1337, params=params)
            clf.load(f)
            pred = clf.predict(X)
            preds.append(pred)

        rf_preds = np.average(preds, axis=0).reshape(-1, 1)

    # predict Extra trees
    with timer('Predicting using extra trees model'):
        params = json_to_dict('models/params/et.json')
        model_files = glob.glob('{}/*.pkl'.format('models/saved_models/et'))
        preds = []
        for f in model_files:
            clf = SklearnWrapper(ExtraTreesRegressor, seed=1337, params=params)
            clf.load(f)
            pred = clf.predict(X)
            preds.append(pred)

        et_preds = np.average(preds, axis=0).reshape(-1, 1)

    ensemble_predictions = np.concatenate((et_preds, rf_preds, lgb_preds, xgb_preds), axis=1)  # note: must retain pred orders

    with timer('Predicting final prediction using ridge regression model'):
        params = json_to_dict('models/params/ridge.json')
        model_files = glob.glob('{}/*.pkl'.format('models/saved_models/ridge'))
        preds = []
        for f in model_files:
            clf = SklearnWrapper(Ridge, seed=1337, params=params)
            clf.load(f)
            pred = clf.predict(ensemble_predictions)
            preds.append(pred)

        final_preds = np.average(preds, axis=0)

    best_threshold = json_to_dict('models/params/general.json')['best_threshold']
    final_preds = (final_preds >= best_threshold).astype(int)

    submission = pd.DataFrame(data={
        'bookingID': feat_aggs['bookingID'],
        'label': final_preds,
    })
    submission.to_csv(write_dest, index=False)

    print('Success:')
    print('submission shape: ', submission.shape)
    print(submission.head())
