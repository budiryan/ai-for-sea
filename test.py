import pandas as pd
import glob

from features.features import RideSafetyFeaturesAggregator

DATA_PATH = 'data/safety/safety'

if __name__ == '__main__':
    features_path = '{}/features'.format(DATA_PATH)
    features = pd.DataFrame()
    for f in glob.glob('{}/*.csv'.format(features_path)):
        temp = pd.read_csv(f)
        print('loaded feature: ', f)
        features = pd.concat([features, temp], axis=0)
        break
    features = features.sort_values(by=['bookingID', 'second'])
    print(features.shape)
    features = features.iloc[:1000]

    feat = RideSafetyFeaturesAggregator(features)
    features_agg = feat.get_aggregated_features()

    print(features_agg.head().T)
    print('feature agg shape: ', features_agg.shape)
