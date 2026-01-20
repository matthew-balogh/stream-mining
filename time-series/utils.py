import numpy as np
from darts import TimeSeries

def apply_window_func(series: TimeSeries, window: int, func):
    """
    Apply a function to each rolling window of a TimeSeries, aligned to the last element.
    
    series: Darts TimeSeries
    window: int, window size
    func: callable, func(window_values, t) -> scalar
          t is the index of the last element in the window
    """

    values = series.values().squeeze()
    times = series.time_index
    results = np.full(len(values), np.nan)

    for t in range(window - 1, len(values)):
        wnd = np.array(values[t - window + 1 : t + 1])
        results[t] = func(wnd, t)
    
    return TimeSeries.from_times_and_values(times, results)
