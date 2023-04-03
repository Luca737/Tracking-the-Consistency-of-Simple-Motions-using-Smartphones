"""File IO stuff for reading and writing into zips, jsons and best axes files"""

import json
import os
import subprocess
from typing import Any, AnyStr, List, Tuple
from zipfile import ZipFile

TRANSFORM_DATA_NAME = "transformedSensData.json"


def readRecordingJSONFromZip(file_path_to_zip: str) -> Tuple[object, str] | None:
    ignore_files = [TRANSFORM_DATA_NAME, "auto_annotated.json", "filteredMotionData.json"]
    with ZipFile(file_path_to_zip, "r") as zip_file:
        for file in zip_file.namelist():
            if not file.lower().endswith(".json"):
                continue
            if any(file.endswith(forbid) for forbid in ignore_files):
                continue
            json_file = zip_file.read(file)
            return json.loads(json_file), file


def readJSONFromZip(file_path_to_zip: str, file_name: str) -> Tuple[object, str] | None:
    with ZipFile(file_path_to_zip, "r") as zip_file:
        for file in zip_file.namelist():
            if not file.endswith(file_name):
                continue
            json_file = zip_file.read(file)
            return json.loads(json_file), file


def saveObjectAsJSONIntoZip(content: object, file_path_to_zip: str, content_file_path: str) -> None:
    already_exists = False
    with ZipFile(file_path_to_zip, "r") as og_zip_file:
        if content_file_path in og_zip_file.namelist():
            already_exists = True
    if already_exists:
        print("File already exists! Overwriting...")
        cmd = ['zip', '-d', file_path_to_zip, content_file_path]
        subprocess.check_call(cmd)
    json_file = json.dumps(content)
    print(content_file_path)
    with ZipFile(file_path_to_zip, "a") as zip_file:
        zip_file.writestr(content_file_path, json_file)


class RecordingsStructureCreator():

    def __init__(self, base_data: dict) -> None:
        self._frames = []
        self._out = dict()
        for key, value in base_data.items():
            self._out[key] = value

    def write_frame(self, frame_attribute: dict, frame_time_stamp: str) -> None:
        self._frames.append({
            "frameStamp": frame_time_stamp,
            "frameAttributes": frame_attribute
        })

    def return_data(self):
        self._out["frames"] = self._frames
        return self._out


# class IntervalStructureCreator():

#     def __init__(self, base_data: dict) -> None:
#         self._intervals = []
#         self._out = dict()
#         for key, value in base_data.items():
#             self._out[key] = value

#     def write_interval(self, interval)

def readBestAxis(file_path: str) -> dict | None:
    if not os.path.isfile(file_path):
        return None
    with open(file_path) as f:
        inp = f.read().split("\n")
    return dict(map(lambda s: s.split("="), inp))
