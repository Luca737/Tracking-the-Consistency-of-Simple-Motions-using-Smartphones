"""Author: Nicola Vidovic

Check the order of the phases in the recording json files

"""

import os

from functions import *
from global_constants import RECORDINGS_PATH

for file_name in os.listdir(RECORDINGS_PATH):
    file = os.path.join(RECORDINGS_PATH, file_name, f"{file_name}.zip")
    if not os.path.isfile(file):
        continue
    raw_data = get_data_from_zip(file_name)
    frames = raw_data["frames"]
    time_stamps = get_time_stamps(frames)

    sorted_time_stamps = sorted(time_stamps)

    if sorted_time_stamps != time_stamps:
        print_red(file_name, "is not sorted")
    else:
        print_green(file_name, "is sorted")
