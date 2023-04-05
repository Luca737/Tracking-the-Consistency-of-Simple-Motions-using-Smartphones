"""Author: Nicola Vidovic

Different filter implementations.

"""

from math import cos
from math import pi as PI
from math import sqrt
from typing import List, Tuple

import global_debug_variables as gdv
import numpy as np
from scipy import ndimage, signal


def median_filter(array: np.array, kernel_size=9) -> np.array:
    """Source: https://github.com/KChen89/Accelerometer-Filtering/blob/master/util/util.py"""
    if gdv.DEBUG_ENABLED:
        if gdv.gdv_median_filter_kernel_size is not None:
            kernel_size = gdv.gdv_median_filter_kernel_size
    shape = array.shape[:2]
    if len(shape) > 2:
        print("Unsupported shape: More then 2 dimensions")
        return
    if len(shape) == 1:
        out = ndimage.median_filter(array, kernel_size)
    elif len(shape) == 2:
        out = np.zeros(shape)
        vector_size = shape[1]
        for part in range(vector_size):
            out[:, part] = ndimage.median_filter(array[:, part], kernel_size)

    return out


def freq_filter(data: np.array, f_size, cutoff) -> np.array:
    """Original Source: https://github.com/KChen89/Accelerometer-Filtering/blob/master/util/util.py
    Low Pass Filter
    """
    data_point, num_signals = data.shape
    f_data = np.zeros([data_point, num_signals])
    lpf = signal.firwin(f_size, cutoff, window='hamming')
    for i in range(num_signals):
        f_data[:, i] = signal.convolve(data[:, i], lpf, mode='same')
    return f_data


def _pass_filter(filter_type: str, data: np.array, cutoff: float | Tuple[float, float], fs: float, order: int = 3) -> np.array:
    """filter_type: 'lowpass' | 'highpass' | 'band'"""
    if gdv.DEBUG_ENABLED:
        if gdv.gdv_pass_filter_order is not None:
            order = gdv.gdv_pass_filter_order
        if filter_type == "lowpass":
            if gdv.gdv_pass_filter_low_cutoff is not None:
                cutoff = gdv.gdv_pass_filter_low_cutoff
        elif filter_type == "highpass":
            if gdv.gdv_pass_filter_high_cutoff is not None:
                cutoff = gdv.gdv_pass_filter_high_cutoff
        elif filter_type == "band":
            if gdv.gdv_pass_filter_high_cutoff is not None and gdv.gdv_pass_filter_low_cutoff is not None:
                cutoff = (gdv.gdv_pass_filter_high_cutoff, gdv.gdv_pass_filter_low_cutoff)
        else:
            print("Wrong filter_type in _pass_filter:", filter_type)

    filtered = np.zeros_like(data)
    b, a = signal.butter(order, cutoff, filter_type, fs=fs)
    if len(data.shape) == 1:
        filtered[:] = signal.filtfilt(b, a, data)
    else:
        num_axis = data.shape[1]
        for i in range(num_axis):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
    return filtered


def low_pass_filter(data: np.array, cutoff: float, fs: float, order: int = 3) -> np.array:
    return _pass_filter("lowpass", data, cutoff, fs, order)


def high_pass_filter(data: np.array, cutoff: float, fs: float, order: int = 3) -> np.array:
    return _pass_filter("highpass", data, cutoff, fs, order)


def band_pass_filter(data: np.array, cutoff: Tuple[float, float], fs: float, order: int = 3) -> np.array:
    return _pass_filter("band", data,  cutoff, fs, order)


def ewma(data: np.array, cutoff: float, fs: float) -> np.array:
    """Exponential Weighted Moving Average"""
    if gdv.DEBUG_ENABLED:
        if gdv.gdv_ewma_weight is not None:
            cutoff = gdv.gdv_ewma_weight
    # Source: https://dsp.stackexchange.com/q/58438
    om = cos(cutoff * PI / (fs/2))
    a = om - 1 + sqrt(om**2 - 4*om + 3)
    # ---------------------------------------------
    out = np.zeros_like(data)
    last_val = np.zeros(data.shape[1])
    for i, point in enumerate(data):
        new_val = (1-a) * last_val + a * point
        last_val = new_val
        out[i] = new_val
    return out


def IIR_low_pass_filter_ord2_single(x: np.array, cutoff: float, fs: float):
    b, a = signal.butter(2, cutoff, "lowpass", fs=fs)
    y = np.zeros_like(x)
    for i in range(2, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] - a[1] * y[i-1] - a[2] * y[i-2]
    y = np.roll(y, -10, axis=0)  # Fix the phase shift
    return y


def IIR_low_pass_filter_ord2_double(x: np.array, cutoff: float, fs: float):
    b, a = signal.butter(2, cutoff, "lowpass", fs=fs)
    y = np.zeros_like(x)
    for i in range(2, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] - a[1] * y[i-1] - a[2] * y[i-2]
    x = np.flip(x, axis=0)
    y = np.flip(y, axis=0)
    for i in range(2, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] - a[1] * y[i-1] - a[2] * y[i-2]
    y = np.flip(y, axis=0)
    # y = np.roll(y, -10, axis=0)
    return y


def IIR_low_pass_filter_ord3_single(x: np.array, cutoff: float, fs: float):
    b, a = signal.butter(3, cutoff, "lowpass", fs=fs)
    y = np.zeros_like(x)
    for i in range(3, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] + b[3] * x[i-3] - a[1] * y[i-1] - a[2] * y[i-2] - a[3] * y[i-3]
    y = np.roll(y, -31, axis=0)
    return y


def IIR_low_pass_filter_ord3_double(x: np.array, cutoff: float, fs: float):
    b, a = signal.butter(3, cutoff, "lowpass", fs=fs)
    y = np.zeros_like(x)
    for i in range(3, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] + b[3] * x[i-3] - a[1] * y[i-1] - a[2] * y[i-2] - a[3] * y[i-3]
    x = np.flip(x, axis=0)
    y = np.flip(y, axis=0)
    for i in range(2, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] + b[3] * x[i-3] - a[1] * y[i-1] - a[2] * y[i-2] - a[3] * y[i-3]
    y = np.flip(y, axis=0)
    # y = np.roll(y, -16, axis=0)
    return y


def IRR_low_pass_filter(x: np.array, cutoff: float, fs: float, order: int, offset: int = 0):
    """Generic IIR butterworth low pass filter."""
    b, a = signal.butter(order, cutoff, "lowpass", fs=fs)
    b = b[::-1]
    a = a[:0:-1]
    if len(x.shape) == 1:
        x = np.concatenate((np.zeros(order), x, np.zeros(order)))
        y = np.zeros(2*order + x.size)
        axes = 1
    else:
        x = np.concatenate((np.zeros((order, x.shape[1])), x, np.zeros((order, x.shape[1]))))
        y = np.zeros((2*order + x.shape[0], x.shape[1]))
        axes = x.shape[1]
    for i in range(order, x.shape[0]):
        for axis in range(axes):
            y[i, axis] = x[:, axis][i-order:i+1].dot(b) - y[:, axis][i-order:i].dot(a)
    # Filter again but in the opposite direction like scipy does.
    # x = np.flip(x, axis=0)
    # y = np.flip(y, axis=0)
    # for i in range(order, x.shape[0]):
    #     for axis in range(axes):
    #         y[i, axis] = x[:, axis][i-order:i+1].dot(b) - y[:, axis][i-order:i].dot(a)
    if offset != 0:
        if len(x.shape) == 1:
            y = np.roll(y, offset)
        else:
            y = np.roll(y, offset, axis=0)
    # y = np.flip(y[order:-order], axis=0)
    return y
