"""Runs the data transformations on all recordings."""

import os

from functions import RECORDINGS_PATH
from transform import transform

for file_name in os.listdir(RECORDINGS_PATH):
    file = os.path.join(RECORDINGS_PATH, file_name, f"{file_name}.zip")
    if not os.path.isfile(file):
        continue
    print(f"----------- {file_name} -----------")
    transform(file_name)
