import glob
import pandas as pd


def load_features_and_labels(features_dir, labels_dir):
    """ Load both ride safety feature data and their labels

    Parameters
    ----------
    features_dir: str
        Path to a folder containing feature CSVs
    labels_dir: str
        Path to a folder containing feature labels

    Returns
    -------
    features: pandas DataFrame
        A pandas DataFrame containing features sorted by (bookingID, second)
    labels: pandas DataFrame
        A pandas DataFrame containing labels
    """
    # load labels
    labels = pd.read_csv(glob.glob('{}/*.csv'.format(labels_dir))[0])
    labels = labels.sort_values(by='bookingID')

    # load features
    features = load_features(features_dir)

    return features, labels


def load_features(features_dir):
    """ Load only ride safety feature data, without their labels

    Parameters
    ----------
    features_dir: str
        Path to a folder containing feature CSVs

    Returns
    -------
    features: pandas DataFrame
        A pandas DataFrame containing features sorted by (bookingID, second)
    """
    features = pd.DataFrame()
    for f in glob.glob('{}/*.csv'.format(features_dir)):
        temp = pd.read_csv(f)
        print('loaded feature file: ', f)
        features = pd.concat([features, temp], axis=0)
    features = features.sort_values(by=['bookingID', 'second'])  # sort by bookingID and chronologically
    return features
