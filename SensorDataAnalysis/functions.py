"""Author: Nicola Vidovic

Various helper functions.

"""

import os
from enum import Enum
from math import sqrt
from operator import itemgetter
from typing import List

import global_debug_variables as gdv
import numpy as np
from FileIO import *
from global_constants import *

# SensorDataType = Enum("SensorDataType", ["linearAcceleration", "gravity", "rotationRate"])


class SensorDataType(Enum):
    linearAcceleration = 0
    gravity = 1
    rotationRate = 2


class AxisType(Enum):
    all = 0
    optimal = 1
    suboptimal = 2


class Axis(Enum):
    x = 0
    y = 1
    z = 2


def print_green(*args):
    print(GREEN[0], end="")
    print(*args, end="")
    print(GREEN[1])


def print_yellow(*args):
    print(YELLOW[0], end="")
    print(*args, end="")
    print(YELLOW[1])


def print_red(*args):
    print(RED[0], end="")
    print(*args, end="")
    print(RED[1])


def is_best_axis_val_valid(val: str) -> bool:
    return ((len(val) == 1 and val in "xyz")
            or (len(val) == 2 and val[0] == "xyz-" and val[1] == "-"))


def get_best_axis_of_type(all_best_axis: dict, data_type: SensorDataType, axis_type: AxisType) -> Axis | None:
    if all_best_axis is None:
        return None
    axis = all_best_axis.get(data_type.name, None)
    if axis == "--" or not is_best_axis_val_valid(axis):
        return None
    if ((axis_type == AxisType.all)
        or (axis_type == AxisType.optimal and len(axis) == 1)
            or (axis_type == AxisType.suboptimal and len(axis) == 2)):
        return Axis[axis[0]]


def get_annotation_from_folder(folder_name: str) -> list | None:
    annotations_file_path = os.path.join(
        RECORDINGS_PATH, folder_name, f"{folder_name}annotations.json")
    if not os.path.isfile(annotations_file_path):
        return

    with open(annotations_file_path) as f:
        json_file = f.read()
    annotations = json.loads(json_file)

    return annotations


def get_data_from_zip(folder_name: str) -> list | None:
    zip_file = os.path.join(RECORDINGS_PATH, folder_name, f"{folder_name}.zip")
    if not os.path.isfile(zip_file):
        return

    data, _ = readRecordingJSONFromZip(zip_file)

    return data


def multi_smooth(array: list) -> list:
    for i in range(6):
        array = smooth_points(array, 4+i)
    return array


def smooth_points(array: np.array, smoothing_radius: int):
    if gdv.DEBUG_ENABLED:
        if gdv.gdv_smooth_points_smoothing_radius is not None:
            smoothing_radius = gdv.gdv_smooth_points_smoothing_radius
    with_padding = np.concatenate([np.repeat(array[0:1], smoothing_radius, axis=0),
                                   array, np.repeat(array[array.shape[0]-1:], smoothing_radius, axis=0)])
    out = []
    for i in range(array.shape[0]):
        out.append(smooth_point(
            with_padding, i+smoothing_radius, smoothing_radius))
    return np.array(out)


def smooth_point(array: np.array, element_index: int, smoothing_radius: int):
    return np.mean(array[element_index-smoothing_radius:element_index+smoothing_radius+1], axis=0)


def slope(index: int, array: list, time_stamps: list) -> float:
    """ Slope between index-1 and index+1.
    Calculate the slope of index in array
    index must be in [1: len(array)-2] (no edges allowed).
    """
    return (array[index+1] - array[index-1]) / (2/100)
    return (array[index+1] - array[index-1]) / (time_stamps[index+1] - time_stamps[index-1])


def l_slope(index: int, array: list, time_stamps: list) -> float:
    """Slope between index-1 and index"""

    return (array[index] - array[index-1]) / (1/100)
    return (array[index] - array[index-1]) / (time_stamps[index] - time_stamps[index-1])


def r_slope(index: int, array: list, time_stamps: list) -> float:
    """Slope between index and index+1"""
    return (array[index+1] - array[index]) / (1/100)
    return (array[index+1] - array[index]) / (time_stamps[index+1] - time_stamps[index])


def derivative(array: np.array, time_stamps: list) -> list:
    out = np.zeros_like(array)
    out[0] = r_slope(0, array, time_stamps)
    for i in range(1, len(array)-1):
        out[i] = slope(i, array, time_stamps)
    out[-1] = l_slope(-1, array, time_stamps)
    return out


def avg(array: list, from_point: int, to_point: int) -> float:
    return sum(array[from_point:to_point]) / (to_point - from_point)


def pos_avg(array: list) -> float:
    points_used = 0
    total = 0
    for point in array:
        if point < 0:
            continue
        total += point
        points_used += 1
    return total / points_used if points_used > 0 else 0

# def pos_avg_square_np() -> np.array:


def pos_avg_np(array: list) -> np.array:
    out = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        out[i] = pos_avg(array[:, i])
    return out


def neg_avg(array: list) -> float:
    points_used = 0
    total = 0
    for point in array:
        if point > 0:
            continue
        total += point
        points_used += 1
    return total / points_used if points_used > 0 else 0


def neg_avg_np(array: list) -> np.array:
    out = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        out[i] = neg_avg(array[:, i])
    return out


def get_secs_from_string(s: str) -> float:
    """Format: HH:MM:ss[.SSSSSS]"""
    hours, mins, secs = s.split(":")
    time_stamp = int(hours) * 3600
    time_stamp += int(mins) * 60
    time_stamp += float(secs)
    return time_stamp


def get_intervals_from_parts(interval_parts: List[dict], time_stamps: list) -> List[Tuple[int, int]]:
    interval_parts.sort(key=itemgetter("start"))
    intervals = []
    start = None
    for interval_part in interval_parts:
        phase = interval_part.get("annotations", {}).get("phase")
        if phase is None:
            print("Interval Part encountered, which has no phase noted:",
                  interval_part["start"], "-", interval_part["end"])
            continue
        phase = int(phase)
        if phase == 0:
            if start is not None:
                print("Last phase had no end. Skipping")
            start = interval_part["start"]
            continue
        if start is None:
            print(f"Phase {phase} encountered with no start:",
                  interval_part["start"], "-", interval_part["end"])
            continue
        if phase == 2:
            start = get_secs_from_string(start)
            start_index = next(i for i, val in enumerate(time_stamps) if val >= start)
            end = get_secs_from_string(interval_part["end"])
            end_index = next(i for i, val in enumerate(time_stamps) if val >= end)
            intervals.append((start_index, end_index))
            start = None

    return intervals


def get_interval_phases_from_parts(interval_parts: List[dict], time_stamps: list) -> List[Tuple[int, int, int, int]]:
    """
    output = [(start_index, stop_phase0_index, stop_phase1_index, stop_phase2_index)]
    """
    interval_parts.sort(key=itemgetter("start"))
    intervals = []
    current_interval = []
    for interval_part in interval_parts:
        phase = interval_part.get("annotations", {}).get("phase")
        if phase is None:
            print_yellow("Warning: Interval Part encountered, which has no phase noted:",
                         interval_part["start"], "-", interval_part["end"])
            continue
        phase = int(phase)
        if phase == 0:
            if len(current_interval) != 0:
                print_yellow("Warning: Start phase encountered - No end for current interval, skipping...",
                             "Time:", interval_part["start"])
                current_interval = []
            current_interval.extend([interval_part["start"], interval_part["end"]])
            continue
        if len(current_interval) == 0:
            print_yellow(f"Warning: Phase {phase} encountered with no start,",
                         "Time:", interval_part["start"])
            continue
        if phase == 1:
            if len(current_interval) != 2:
                print_yellow("Warning: Phase 1 encountered while not valid (two phase 1?)",
                             "Time:", interval_part["start"])
                current_interval = []
                continue
            current_interval.append(interval_part["end"])
            continue
        if phase == 2:
            if len(current_interval) == 2:
                # No phase 1 (pause) -> stop is stop of phase 0.
                current_interval.append(current_interval[1])
            current_interval.append(interval_part["end"])
            current_interval = map(get_secs_from_string, current_interval)
            # Converting each time to it's corresponding time_point_index in time_stamps.
            current_interval = map(
                lambda time_val: next(i for i, val in enumerate(time_stamps) if val >= time_val),
                current_interval
            )
            intervals.append(tuple(current_interval))
            current_interval = []

    return intervals


class StatisticsInterval():

    def __init__(self) -> None:
        self.start_time = None
        self.start_time_index = None
        self.end_time = None
        self.end_time_index = None
        self.extrema = None
        self.extrema_indexes = None
        self.average = None
        self.is_parabolic = None  # Not about it being symmetric (only one extremum; no platos)

    def __str__(self) -> str:
        sign = "+" if self.average == abs(self.average) else "-"
        out = f"  {round(self.start_time, 3)}-{round(self.end_time, 3)} ({round(self.end_time-self.start_time, 3)}): {sign} "
        out += f"avg: {abs(self.average)}, extrema: "
        for extremum in self.extrema:
            out += f"{abs(round(extremum, 6))}, "
        return out


def calc_statistics(time_stamps: list, start: int, stop: int, zero_points: list, values: list) -> List[StatisticsInterval]:
    sections = []
    section_stop = start
    while section_stop < stop:
        section = StatisticsInterval()
        section_start = section_stop
        section_stop = next((i for i in zero_points if i > section_start), None)
        if section_stop is None or section_stop > stop:
            section_stop = stop
        section.start_time = time_stamps[section_start]
        section.start_time_index = section_start
        section.end_time = time_stamps[section_stop]
        section.end_time_index = section_stop

        # Average.
        section.average = avg(values, section_start, section_stop)

        # Extrema.
        section.extrema = []
        section.extrema_indexes = []
        last_point = values[section_start]
        is_rising = True
        for i, point in enumerate(values[section_start+1:section_stop], section_start+1):
            if abs(point) >= abs(last_point):
                is_rising = True
                last_point = point
                continue
            if is_rising:
                section.extrema.append(last_point)
                section.extrema_indexes.append(i)
            is_rising = False
            last_point = point

        # Parabolic.
        section.is_parabolic = len(section.extrema) == 1

        sections.append(section)

    return sections


def print_statistics_from_intervals(intervals: list, data: list, time_stamps: list, zero_points: list) -> None:
    for start, stop in intervals:
        print("Interval:", round(time_stamps[start], 3), "-", round(time_stamps[stop], 3),
              f"({round(time_stamps[stop]-time_stamps[start], 3)})")
        interval_statistics = calc_statistics(
            time_stamps, start, stop, zero_points, data)
        for statistic in interval_statistics:
            print(statistic)


def get_motion_data_interval(raw_intervals: List[dict], time_stamps: list, offset: float = MOTION_INTERVAL_OFFSET_DEFAULT):
    """Returns the start and stop time point index of the actual motion recording
    of the exercise (-/+ the offset) -> Movements at the start and stop of the recording, that
    are just for getting into position or stopping the recording are removed."""

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_get_motion_data_interval_offset is not None:
            offset = gdv.gdv_get_motion_data_interval_offset

    start = min(
        get_secs_from_string(interval["start"]) for interval in raw_intervals
    ) - offset
    stop = max(
        get_secs_from_string(interval["end"]) for interval in raw_intervals
    ) + offset
    start = next(i for i, time_point in enumerate(time_stamps) if time_point >= start)
    start = max(start - 1, 0)
    stop = next((i for i, time_point in enumerate(time_stamps) if time_point >= stop), None)
    if stop is None:
        stop = len(time_stamps) - 1
    return (start, stop)


def calc_key_features(values: list, time_stamps: list) -> list:
    """
    Features:
        - Simplify to positive, negative and zero
        - Calc zero points (sign changed compared to the prior one)
        - Zero plato ranges (Continuous range where all points are classified as 0) range is inclusivefu
    """
    zero_plato_val_radius = 0.01
    zero_plato_slope_radius = 0.1

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_zero_plato_val_radius is not None:
            zero_plato_val_radius = gdv.gdv_zero_plato_val_radius
        if gdv.gdv_zero_plato_slope_radius is not None:
            zero_plato_slope_radius = gdv.gdv_zero_plato_slope_radius

    pos_neg_zero = []
    # Indexes of points where the sign changed.
    switch_sign_points = []
    last_sign = values[1] >= 0
    for i, val in enumerate(values[1:len(values)-1], 1):
        sign = val >= 0
        if sign != last_sign:
            last_sign = sign
            switch_sign_points.append(i)
        if val > zero_plato_val_radius:
            cur_pos_neg_zero = 0.25
        elif val < -zero_plato_val_radius:
            cur_pos_neg_zero = -0.25
        elif -zero_plato_slope_radius <= slope(i, values, time_stamps) <= zero_plato_slope_radius:
            cur_pos_neg_zero = 0
        elif val >= 0:
            cur_pos_neg_zero = 0.25
        else:
            cur_pos_neg_zero = -0.25
        pos_neg_zero.append(cur_pos_neg_zero)
    pos_neg_zero.insert(0, pos_neg_zero[0])
    pos_neg_zero.append(pos_neg_zero[-1])

    # Get zero plato's.
    zero_platos = []  # = [(start, stop), ...] -> [start, stop)
    start, stop = None, None
    in_plato = False
    for i, val in enumerate(pos_neg_zero):
        if not in_plato and val == 0:
            start = i
            in_plato = True
        elif in_plato and val != 0:
            stop = i-1
            zero_platos.append((start, stop))
            in_plato = False
    if in_plato:
        stop = i
        zero_platos.append((start, stop))

    return pos_neg_zero, switch_sign_points, zero_platos


def calc_zero_points_platos(data: list, data_derivative: list, zero_plato_val_radius: float, zero_plato_slope_radius: float, min_plato_length_secs: float, allowed_skip_length_secs: float) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Returns:
        - Zero points: Point at which the sign changed to the point prior.
        - Zero Platos: Ranges of connected points, which are determined to be close to zero (Absolute value close to zero and derivative)
            -> Ranges: [start, stop] -> stop is inclusive.
    """

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_zero_plato_val_radius is not None:
            zero_plato_val_radius = gdv.gdv_zero_plato_val_radius
        if gdv.gdv_zero_plato_slope_radius is not None:
            zero_plato_slope_radius = gdv.gdv_zero_plato_slope_radius
        if gdv.gdv_zero_plato_min_length is not None:
            min_plato_length_secs = gdv.gdv_zero_plato_min_length
        if gdv.gdv_zero_plato_allowed_skip_length is not None:
            allowed_skip_length_secs = gdv.gdv_zero_plato_allowed_skip_length

    min_plato_length = round(min_plato_length_secs * FS)
    max_gap_length_to_bridge = round(allowed_skip_length_secs * FS)

    zero_platos = []
    zero_points = []
    last_sign = data[1] >= 0
    plato_start = None
    for i, val in enumerate(data[1:len(data)-1], 1):
        sign = val >= 0
        if sign != last_sign:
            last_sign = sign
            zero_points.append(i)
        is_valid_zero_plato_point = (abs(val) <= zero_plato_val_radius) and (abs(data_derivative[i]) <= zero_plato_slope_radius)
        if (plato_start is not None) and (not is_valid_zero_plato_point):
            zero_platos.append((plato_start, i-1))
            plato_start = None
        elif (plato_start is None) and (is_valid_zero_plato_point):
            plato_start = i

    if plato_start is not None:
        zero_platos.append((plato_start, len(data)-1))

    # Connecting platos with an edge distance of at most max_gap_length_to_bridge.
    for i in range(len(zero_platos)-1, 0, -1):
        if zero_platos[i][0] - zero_platos[i-1][1] <= max_gap_length_to_bridge:
            zero_platos[i-1] = (zero_platos[i-1][0], zero_platos[i][1])
            del zero_platos[i]

    # Removing non significant platos.
    for i in range(len(zero_platos)-1, -1, -1):
        if zero_platos[i][1] - zero_platos[i][0] < min_plato_length:
            del zero_platos[i]

    return zero_points, zero_platos


def calc_zero_platos(data: list, data_derivative: list) -> List[Tuple[int, int]]:
    """Ranges of zero_platos are inclusive -> [start, stop]"""
    zero_plato_val_radius = 0.20
    zero_plato_slope_radius = 0.20 * 2

    # Minimum length to be significant (for non zero and zero platos).
    min_plato_length_secs = 1/10
    allowed_skip_length_secs = 1/10

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_zero_plato_val_radius is not None:
            zero_plato_val_radius = gdv.gdv_zero_plato_val_radius
        if gdv.gdv_zero_plato_slope_radius is not None:
            zero_plato_slope_radius = gdv.gdv_zero_plato_slope_radius
        if gdv.gdv_zero_plato_min_length is not None:
            min_plato_length_secs = gdv.gdv_zero_plato_min_length
        if gdv.gdv_zero_plato_allowed_skip_length is not None:
            allowed_skip_length_secs = gdv.gdv_zero_plato_allowed_skip_length

    min_plato_length = round(min_plato_length_secs * FS)
    max_gap_length_to_bridge = round(allowed_skip_length_secs * FS)

    data = np.array(data)
    data_derivative = np.array(data_derivative)

    zero_points = (
        (data >= -zero_plato_val_radius)
        & (data <= zero_plato_val_radius)
        & (data_derivative >= -zero_plato_slope_radius)
        & (data_derivative <= zero_plato_slope_radius)
    ).astype(int)
    starts_and_stops = np.diff(zero_points)
    starts = list(np.where(starts_and_stops == 1)[0] + 1)
    stops = list(np.where(starts_and_stops == -1)[0])

    if zero_points[0] == 1:
        starts.insert(0, 0)
    if zero_points[-1] == 1:
        stops.append(data.size - 1)

    # Connecting platos with under significant separation.
    for i in range(len(starts)-1, 0, -1):
        if starts[i] - stops[i-1] <= max_gap_length_to_bridge:
            del starts[i]
            del stops[i-1]

    # Removing non significant platos.
    for i in range(len(starts)-1, -1, -1):
        if stops[i] - starts[i] < min_plato_length:
            del starts[i]
            del stops[i]

    assert len(starts) == len(stops)
    return list(zip(starts, stops))


def calc_platos(data_derivative: list, slope_significance: float = 0.09, min_plato_length_secs: float = 0.2, allowed_skip_length_secs: float = 0.1) -> List[Tuple[int, int]]:
    data_derivative = np.array(data_derivative)
    zero_points = (
        (data_derivative > -slope_significance)
        & (data_derivative < slope_significance)
    ).astype(int)
    starts_and_stops = np.diff(zero_points)
    starts = list(np.where(starts_and_stops == 1)[0] + 1)
    stops = list(np.where(starts_and_stops == -1)[0])

    if zero_points[0] == 1:
        starts.insert(0, 0)
    if zero_points[-1] == 1:
        stops.append(data_derivative.size - 1)

    # Minimum length to be significant (for non zero and zero platos).
    min_plato_length = round(min_plato_length_secs * FS)
    max_gap_length_to_bridge = round(allowed_skip_length_secs * FS)

    # Connecting platos with under significant separation.
    for i in range(len(starts)-1, 0, -1):
        if starts[i] - stops[i-1] < max_gap_length_to_bridge:
            del starts[i]
            del stops[i-1]

    # Removing non significant platos.
    for i in range(len(starts)-1, -1, -1):
        if stops[i] - starts[i] < min_plato_length:
            del starts[i]
            del stops[i]

    assert len(starts) == len(stops)
    return list(zip(starts, stops))


def get_attributes(frames: list, attr: list) -> np.array:
    return np.array(list(
        map(itemgetter(*attr),
            map(itemgetter("frameAttributes"), frames)))
    )


def get_gravity(frames: list) -> np.array:
    return get_attributes(frames, ["gravity_X", "gravity_Y", "gravity_Z"])


def get_acceleration(frames: list) -> np.array:
    return get_attributes(frames, ["acceleration_X", "acceleration_Y", "acceleration_Z"])


def get_rotation_rate(frames: list) -> np.array:
    if "rotationRate_X" in frames[0]["frameAttributes"]:
        return get_attributes(frames, ["rotationRate_X", "rotationRate_Y", "rotationRate_Z"])


def get_time_stamps(frames: list) -> list:
    time_stamps = []
    for frame in frames:
        time_stamp = get_secs_from_string(frame["frameStamp"])
        time_stamps.append(time_stamp)
    return time_stamps


def auto_choose_axis(all_axes_data: np.array) -> Axis:
    all_axes_data = np.abs(all_axes_data)
    total_avg = np.average(all_axes_data, axis=0)
    axis = max(enumerate(total_avg), key=itemgetter(1))[0]

    return Axis(axis)


def auto_choose_axis1(all_axes_data: np.array) -> Axis:
    averages = []
    for i in range(3):
        axis_data = all_axes_data[:, i]
        minima_baseline = min(axis_data) / 2
        maxima_baseline = max(axis_data) / 2
        minima, maxima = calc_extrema_with_baseline(axis_data, minima_baseline, maxima_baseline)
        averages.append((abs(sum(map(lambda x: axis_data[x], minima))) + sum(map(lambda x: axis_data[x], maxima))) / (len(minima)+len(maxima)) if (len(minima)+len(maxima)) > 0 else 0)
    axis = max(enumerate(averages), key=itemgetter(1))[0]

    return Axis(axis)


def auto_swap(array: np.array) -> None:
    max_pc0 = np.max(array[:, 0])
    min_pc0 = np.min(array[:, 0])
    max_pc1 = np.max(array[:, 1])
    min_pc1 = np.min(array[:, 1])
    swap = None
    if max_pc0 > max_pc1 and min_pc0 < min_pc1:
        swap = False
    elif max_pc0 < max_pc1 and min_pc0 < min_pc1:
        swap = True
    elif max_pc0 > max_pc1:
        if min_pc0 == min_pc1:
            swap = False
        # min_pc0 > min_pc1
        elif abs(min_pc0) >= max_pc1:
            swap = False
        else:
            swap = True
    elif max_pc0 < max_pc1:
        if min_pc0 == min_pc1:
            swap = True
        elif max_pc0 <= abs(min_pc1):
            swap = True
        else:
            swap = False
    elif min_pc0 < min_pc1:
        if max_pc0 == max_pc1:
            swap = False
        elif max_pc0 >= abs(min_pc1):
            swap = False
        else:
            swap = True
    elif min_pc0 > min_pc1:
        if max_pc0 == max_pc1:
            swap = True
        elif abs(min_pc0) <= max_pc1:
            swap = True
        else:
            swap = False

    if swap is None:
        print("Can't make out if PC's have to be swapped: Implement a better method")

    if swap:
        buffer = np.copy(array[:, 0])
        array[:, 0] = array[:, 1]
        array[:, 1] = buffer
        print("\n  Swapped PC's\n")


def calc_extrema_with_baseline(data: list, minima_baseline: float, maxima_baseline: float) -> Tuple[List[int], List[int]]:
    """Return: (minima, maxima)"""
    minima = []
    maxima = []
    min_i = 0
    min_val = 0
    max_i = 0
    max_val = 0
    for i, val in enumerate(data):
        if val <= minima_baseline:
            if val < min_val:
                min_val = val
                min_i = i
        elif min_val < 0:
            minima.append(min_i)
            min_val = 0
        if val >= maxima_baseline:
            if val > max_val:
                max_val = val
                max_i = i
        elif max_val > 0:
            maxima.append(max_i)
            max_val = 0

    return minima, maxima


def plot_3d(data: np.array):
    from math import cos, pi, sin

    import matplotlib.pyplot as plt

    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X, Y, Z)

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    radius = 80

    elev = 20
    roll = 0
    # Rotate the axes and update
    for angle in range(0, 360*4 + 1, 5):
        # Normalize the angle to the range [-180, 180] for display
        # angle_norm = (angle + 180) % 360 - 180

        x = radius * cos(angle * pi / 180)
        y = radius * sin(angle * pi / 180)
        z = 0

        # Rotate around x-axis:
        y = y*cos(45 * pi / 180)
        z = y*sin(45 * pi / 180)

        # Rotate around y-axis:
        x = x*cos(45 * pi / 180) + z*sin(45 * pi / 180)
        z = x*sin(45 * pi / 180) - z*cos(45 * pi / 180)

        elev, azim, roll = x, y, z

        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

        plt.draw()
        plt.pause(.001)


def print_conditional(do_print: bool, *args) -> None:
    if do_print:
        print(*args)


def print_results(sums: Tuple[int, int, float, float, float, float], total_truths: int, n_files: int, with_phases: bool) -> None:
    if with_phases:
        (
            total_truths_matched,
            total_false_or_duplicates,
            total_length_error,
            total_phase0_error,
            total_phase1_error,
            total_phase2_error,
            total_length_error_percent,
            total_phase0_error_percent,
            total_phase1_error_percent,
            total_phase2_error_percent,
            std_length_error,
            std_phase0_error,
            std_phase1_error,
            std_phase2_error,
            std_length_error_percent,
            std_phase0_error_percent,
            std_phase1_error_percent,
            std_phase2_error_percent,
            mean_length_error,
            mean_phase0_error,
            mean_phase1_error,
            mean_phase2_error,
            mean_length_error_percent,
            mean_phase0_error_percent,
            mean_phase1_error_percent,
            mean_phase2_error_percent
        ) = sums
    else:
        (
            total_truths_matched,
            total_false_or_duplicates,
            total_length_error,
            total_length_error_percent,
            std_length_error,
            std_length_error_percent,
            mean_length_error,
            mean_length_error_percent
        ) = sums
    precision = total_truths_matched / (total_truths_matched + total_false_or_duplicates) if total_truths_matched + total_false_or_duplicates > 0 else 0
    recall = total_truths_matched / total_truths
    f_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    print(f"  Precision      | {round(precision*100, 3):<10}")
    print(f"  Recall         | {round(recall*100, 3):<10}")
    print(f"  F-Score        | {round(f_score*100, 3):<10}")
    print(f"  Correctness    | {f'{total_truths_matched}/{total_truths+total_false_or_duplicates}':<10} accuracy: {round(total_truths_matched/(total_truths+total_false_or_duplicates)*100, 3)}")
    print(f"  Truths Matched | {f'{total_truths_matched}/{total_truths}':<10} accuracy: {round(total_truths_matched/total_truths*100, 3)}")
    print(f"  False          | {total_false_or_duplicates:<10} avg: {round(total_false_or_duplicates/total_truths_matched, 3) if total_truths_matched > 0 else '-'}")
    print(f"  Length Error   | {round(total_length_error, 3):<10} avg: {round(total_length_error/total_truths_matched if total_truths_matched > 0 else 0, 3):<6} ± {round(std_length_error if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_length_error, 3)}")
    if with_phases:
        print(f"  Phase0 Error   | {round(total_phase0_error, 3):<10} avg: {round(total_phase0_error/total_truths_matched if total_truths_matched > 0 else 0, 3):<6} ± {round(std_phase0_error if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase0_error, 3)}")
        print(f"  Phase1 Error   | {round(total_phase1_error, 3):<10} avg: {round(total_phase1_error/total_truths_matched if total_truths_matched > 0 else 0, 3):<6} ± {round(std_phase1_error if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase1_error, 3)}")
        print(f"  Phase2 Error   | {round(total_phase2_error, 3):<10} avg: {round(total_phase2_error/total_truths_matched if total_truths_matched > 0 else 0, 3):<6} ± {round(std_phase2_error if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase2_error, 3)}")
    print("  --------------------------------------")
    print(f"  Length Error % | {round(total_length_error_percent*100, 3):<10} avg: {round((total_length_error_percent/total_truths_matched if total_truths_matched > 0 else 0)*100, 3):<6} ± {round(std_length_error_percent*100 if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_length_error_percent*100, 3)}")
    if with_phases:
        print(f"  Phase0 Error % | {round(total_phase0_error_percent*100, 3):<10} avg: {round((total_phase0_error_percent/total_truths_matched if total_truths_matched > 0 else 0)*100, 3):<6} ± {round(std_phase0_error_percent*100 if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase0_error_percent*100, 3)}")
        print(f"  Phase1 Error % | {round(total_phase1_error_percent*100, 3):<10} avg: {round((total_phase1_error_percent/total_truths_matched if total_truths_matched > 0 else 0)*100, 3):<6} ± {round(std_phase1_error_percent*100 if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase1_error_percent*100, 3)}")
        print(f"  Phase2 Error % | {round(total_phase2_error_percent*100, 3):<10} avg: {round((total_phase2_error_percent/total_truths_matched if total_truths_matched > 0 else 0)*100, 3):<6} ± {round(std_phase2_error_percent*100 if total_truths_matched > 0 else 0, 3):<8} median: {round(mean_phase2_error_percent*100, 3)}")
    print()
