import numpy as np
import pandas as pd


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  group=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0,
                  measure='mean'):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    # assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=group)[target.name].agg([measure, "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages[measure] * smoothing
    averages.drop([measure, "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    if isinstance(group, list) and len(group) >= 2:
        # print(group)
        ft_trn_series = pd.merge(
            trn_series,
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=group,
            how='left')['average'].rename(('').join(group) + '_' + measure).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index

        ft_tst_series = pd.merge(
            tst_series,
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=group,
            how='left')['average'].rename(('').join(group) + '_' + measure).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
    else:
        # print(group)
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_' + measure).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index

        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_' + measure).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
