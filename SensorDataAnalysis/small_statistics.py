"""Author: Nicola Vidovic

Used for producing some smaller statistics for easier fine tuning of methods.
Note that this has to be eddied by hand to produce different statistics.

"""

import os

from filters import *
from functions import *
from global_constants import RECORDINGS_PATH

all_minima = []
all_maxima = []
max_slopes = []
max_vals = []
for file_name in os.listdir(RECORDINGS_PATH):
    file = os.path.join(RECORDINGS_PATH, file_name, f"{file_name}.zip")
    if not os.path.isfile(file):
        continue
    raw_data = get_data_from_zip(file_name)
    annotations = get_annotation_from_folder(file_name)
    if raw_data is None or annotations is None:
        continue
    best_axis = get_best_axis_of_type(readBestAxis(os.path.join(RECORDINGS_PATH, file_name, AUTO_BEST_AXIS_FILE_NAME)), SensorDataType.rotationRate, AxisType.optimal)
    if best_axis is None:
        continue
    frames = raw_data["frames"]
    time_stamps = get_time_stamps(frames)
    interval_parts = annotations["intervals"]
    global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)
    time_stamps = time_stamps[global_start:global_stop+1]

    data = get_rotation_rate(frames)
    if data is None:
        continue
    axis_data = low_pass_filter(data[global_start:global_stop+1, best_axis.value], 2.5, 100)
    # data = derivative(data, time_stamps)

    max_slopes.append(max(np.abs(derivative(axis_data, time_stamps))))
    max_vals.append(max(np.abs(axis_data)))

    # axis_data = data[:, best_axis.value]
    minima, maxima = calc_extrema_with_baseline(axis_data, min(axis_data) / 2, max(axis_data) / 2)
    minima = list(map(lambda x: axis_data[x], minima))
    maxima = list(map(lambda x: axis_data[x], maxima))
    if min(maxima) <= 0.09 or max(minima) >= -0.11:
        print("----", file_name, "----")
        print(max(minima))
        print(min(maxima))
    all_minima.extend(minima)
    all_maxima.extend(maxima)

print("Maximum of Minima:", max(all_minima), "; min:", min(all_minima))
print("Minimum of Maxima:", min(all_maxima), "; max:", max(all_maxima))

print(f"Slope: max {max(max_slopes)}, min {min(max_slopes)}, avg {avg(max_slopes, 0, len(max_slopes))}")
print(f"Vals: max {max(max_vals)}, min {min(max_vals)}, avg {avg(max_vals, 0, len(max_vals))}")
