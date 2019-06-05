import pandas as pd
import glob

from features.features import RideSafetyFeaturesAggregator
from dataloader.dataloader import load_ride_safety_data

DATA_PATH = 'data/safety/safety'

if __name__ == '__main__':
    DATA_PATH = 'data/safety/safety'

    features, labels = load_ride_safety_data('{}/features'.format(DATA_PATH), '{}/labels'.format(DATA_PATH))

    feat = RideSafetyFeaturesAggregator(features)
    features_agg = feat.get_aggregated_features()

    print(features_agg.head().T)
    print('feature agg shape: ', features_agg.shape)
