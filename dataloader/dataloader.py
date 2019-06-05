import glob
import pandas as pd


def load_ride_safety_data(features_dir, labels_dir):
    # load labels
    labels = pd.read_csv(glob.glob('{}/*.csv'.format(labels_dir))[0])
    labels = labels.sort_values(by='bookingID')

    # load features
    features = pd.DataFrame()
    for f in glob.glob('{}/*.csv'.format(features_dir)):
        temp = pd.read_csv(f)
        print('loaded feature: ', f)
        features = pd.concat([features, temp], axis=0)
    features = features.sort_values(by=['bookingID', 'second'])  # sort by bookingID and chronologically

    return features, labels
