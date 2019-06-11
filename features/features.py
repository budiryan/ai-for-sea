import pandas as pd
import numpy as np
import itertools


class RideSafetyFeaturesAggregator(object):
    """
    A class generating various statistical features and then aggregates them by their respective bookingIDs.
    Various helper functions are provided, using `get_aggregated_features` is enough

    Parameters
    ----------
    input_df: pandas.DataFrame()
        A DataFrame containing features from Grab's ride safety dataset
    """
    def __init__(self, input_df):
        self.ride_safety_df = input_df

    def get_aggregated_features(self):
        """
        Get various statistical and aggregation features.
        TODO: explain the various features

        Returns
        -------
        out_df: pandas.DataFrame()
            An aggregated features DataFrame aggregated by bookingID
        """

        temp = self._aggregate_basic_statistical_features()  # get basic statistical features

        other_features = self.ride_safety_df.groupby('bookingID', as_index=True).apply(self._aggregate_special_features)
        other_features = other_features.reset_index()
        return pd.merge(temp, other_features, how='left', on='bookingID')

    def _aggregate_basic_statistical_features(self):
        def percentile25(x):
            return x.quantile(0.25)

        def percentile50(x):
            return x.median()

        def percentile75(x):
            return x.quantile(0.75)

        aggregate_functions = ['mean', 'min', 'max', 'std', percentile25, percentile50, percentile75]
        agg_columns_excluded = ['bookingID', 'second']
        agg_dict = {c: aggregate_functions for c in self.ride_safety_df.columns if c not in agg_columns_excluded}
        agg_dict['second'] = ['max']

        agg_df = self.ride_safety_df.groupby(['bookingID'], as_index=True).agg(agg_dict)
        agg_df.columns = agg_df.columns.map('_'.join)
        agg_df = agg_df.reset_index(drop=False)

        return agg_df

    def _aggregate_special_features(self, inp):
        n_stops, hit_mean, hit_max, hit_std = self._get_stopping_statistics(inp)
        naive_dist = self._get_naive_distance(inp)
        stopping_time_ratio = self._get_relative_stopping_time(inp)
        num_acceleration_change_x = len(list(itertools.groupby(inp['acceleration_x'], lambda x: x > 0))) / inp['second'].max()
        num_acceleration_change_y = len(list(itertools.groupby(inp['acceleration_y'], lambda x: x > 0))) / inp['second'].max()
        num_acceleration_change_z = len(list(itertools.groupby(inp['acceleration_z'], lambda x: x > 0))) / inp['second'].max()

        d = {
            'n_stops': n_stops,
            'hit_mean': hit_mean,
            'hit_max': hit_max,
            'hit_std': hit_std,
            'naive_distance': naive_dist,
            'stopping_time_ratio': stopping_time_ratio,
            'num_acceleration_change_x': num_acceleration_change_x,
            'num_acceleration_change_y': num_acceleration_change_y,
            'num_acceleration_change_z': num_acceleration_change_z,
        }
        return pd.Series(d, index=['n_stops', 'hit_mean', 'hit_max', 'hit_std',
                                   'naive_distance', 'stopping_time_ratio', 'num_acceleration_change_x',
                                   'num_acceleration_change_y', 'num_acceleration_change_z'])

    @staticmethod
    def _get_naive_distance(inp):
        return ((inp['second'].shift(-1) - inp['second']).fillna(0) * inp['Speed']).sum()

    @staticmethod
    def _get_relative_stopping_time(inp):
        last = round(len(inp) * 0.05)
        eps = 1

        # determine the stopping ratio of last 5% of a trip
        speed_red = inp.Speed.values[len(inp.Speed.values) - last:]
        return len(speed_red[speed_red < 0 + eps]) / float(inp.second.max())

    @staticmethod
    def _get_stopping_statistics(inp):
        # gets every vehicle stop in a trip and returns its start_time, end_time and diff

        # make sure all runs of ones are well-bounded
        bounded = np.hstack(([1], inp.Speed.values, [1]))

        log = (bounded < 0.5) * 1

        # get 1 at run starts and -1 at run ends
        diffs = np.diff(log)

        # get indices if starts and ends
        run_starts = np.where(diffs > 0)[0]
        run_ends = np.where(diffs < 0)[0]

        interval = 7
        end_stops = np.array([run_starts, run_ends, run_ends - run_starts]).T
        end_stops = end_stops.astype(int)[:-1, 1]
        end_stops = end_stops[end_stops + interval < len(inp.Speed.values) - 1]

        n_stops = len(end_stops)

        if n_stops > 1:
            hit = np.zeros(shape=(1, n_stops))
            for i in range(n_stops):
                # slope at acceleration
                start = end_stops[i]
                hit[0, i] = np.diff([inp.Speed.values[start], inp.Speed.values[start + interval]])
        else:
            hit = np.array([0])

        return [n_stops, hit.mean(), hit.max(), hit.std()]

