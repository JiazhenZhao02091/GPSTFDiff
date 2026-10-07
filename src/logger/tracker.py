import pandas as pd
import torch
import numpy as np

eps = 1e-16


class Tracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(
            0.0,
            index=keys, columns=['total', 'counts', 'average', 'now']
        )
        self.reset()

    def reset(self):
        self._data.loc[:, :] = 0.0

    def update(self, key, value, n=1):
        if np.isnan(value):
            return


        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        self._data.loc[key, "now"] = value

    def avg(self, key):
        return self._data.average[key] if self._data.counts[key] > 0 else np.nan

    @property
    def results(self):
        return dict(self._data.average.where(self._data.counts > 0, np.nan))

    @property
    def now(self):
        return dict(self._data.now)
