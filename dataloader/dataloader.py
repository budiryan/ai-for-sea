import glob
import pandas as pd


def load_ride_safety_train(features_dir, labels_dir):
    # load labels
    labels = pd.read_csv(glob.glob('{}/*.csv'.format(labels_dir))[0])
    labels = labels.sort_values(by='bookingID')

    # load features
    features = _load_features(features_dir)

    return features, labels


def load_ride_safety_test(features_dir):
    # load features
    features = _load_features(features_dir)
    return features


def _load_features(features_dir):
    features = pd.DataFrame()
    for f in glob.glob('{}/*.csv'.format(features_dir)):
        temp = pd.read_csv(f)
        print('loaded feature file: ', f)
        features = pd.concat([features, temp], axis=0)
    features = features.sort_values(by=['bookingID', 'second'])  # sort by bookingID and chronologically
    return features
