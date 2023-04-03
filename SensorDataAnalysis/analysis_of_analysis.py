"""
Simple brute forcing of the parameters for that the analysis methods use (for zero platos, zero points, extrema, ...).
However this is too slow, without optimizing the loading of the motion data files and more. While I
also wanted to use hill climbing I decided against it, as trying by hand was simple and way faster.

"""

import os
import pickle
from itertools import product
from time import time

import global_debug_variables as gdv
from analyze import analyze_rot_rate_v2_1
from analyze_truth import rate_analysis
from functions import SensorDataType
from global_constants import FS

FOLDER_PATH = "ValuesOfRateAnalysisOutput"
OUT_FILE_NAME = "rotRateV2_1" + ".csv"
FILE_PATH = os.path.join(FOLDER_PATH, OUT_FILE_NAME)
if os.path.isfile(FILE_PATH):
    print(f"'{OUT_FILE_NAME}' already exists")
    exit()


def frange(start, stop, step=1):
    out = []
    i = 0
    val = round(start+step*i, 4)
    while val <= stop:
        out.append(val)
        i += 1
        val = round(start+step*i, 4)
    return out


analysis_method = analyze_rot_rate_v2_1
sens_type = SensorDataType.rotationRate

minima_baseline = maxima_baseline = list(frange(1.4, 3, 0.2))
plato_val_radius = list(frange(0.06, 0.40, 0.02))
plato_slope_radius = list(frange(0.10, 0.80, 0.05))
plato_min_length = list(frange(7, 20, 1))
plato_skip_length = list(frange(5, 18, 1))

settings = product(minima_baseline, maxima_baseline, plato_val_radius, plato_slope_radius, plato_min_length, plato_skip_length)

total_interactions_to_make = len(minima_baseline) * len(maxima_baseline) * len(plato_val_radius) * len(plato_slope_radius) * len(plato_min_length) * len(plato_skip_length)
iterations_made = 0
block_size = 10
start_time = time()
with open(FILE_PATH, "ab") as f:
    for setting in settings:
        gdv.gdv_analysis_minima_baseline_factor = setting[0]
        gdv.gdv_analysis_maxima_baseline_factor = setting[1]
        gdv.gdv_zero_plato_val_radius = setting[2]
        gdv.gdv_zero_plato_slope_radius = setting[3]
        gdv.gdv_zero_plato_min_length = setting[4]
        gdv.gdv_zero_plato_allowed_skip_length = setting[5]

        result = rate_analysis(analysis_method, sens_type, True, verbose=False)

        pickle.dump((
            {
                "minima_baseline": setting[0],
                "maxima_baseline": setting[1],
                "plato_val_radius": setting[2],
                "plato_slope_radius": setting[3],
                "plato_min_length": setting[4],
                "plato_skip_length": setting[5]
            },
            result
        ), f)

        iterations_made += 1
        if iterations_made % block_size == 0:
            stop_time = time()
            time_remaining = (total_interactions_to_make-iterations_made) * (stop_time-start_time) / block_size
            pure_seconds, minutes = time_remaining % 60, time_remaining // 60
            pure_minutes, pure_hours = minutes % 60, minutes // 60
            print(f"{iterations_made}/{total_interactions_to_make} - Estimated Time: {round(pure_hours)}:{round(pure_minutes)}:{round(pure_seconds)}")
            start_time = time()
