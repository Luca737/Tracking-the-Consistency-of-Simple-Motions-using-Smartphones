"""Author: Nicola Vidovic"""

import os
import subprocess
import threading
from datetime import datetime
from enum import Enum

from connection import Connection
from network_browser import NetworkBrowser
from video_capture import VideoCapture


class Command(Enum):
    heartbeat = 0
    start = 1
    stop = 2
    data = 3
    enableVideoModule = 4


class NetworkControllerDelegate:

    def start_response(self) -> None:
        return

    def stop_response(self) -> None:
        return

    def data_response(self, data: bytes) -> None:
        return

    def enable_video_recording_module(self) -> None:
        return


class NetworkController:

    def __init__(self, connection: Connection, delegate: NetworkControllerDelegate) -> None:
        self._connection = connection
        self.delegate = delegate
        self._controller_thread = None
        self._active = False

    def start(self) -> None:
        self._connection.start()
        self._controller_thread = threading.Thread(
            target=self._run, name="connection")
        self._controller_thread.start()

    def _run(self) -> None:
        self._active = True
        while self._active:
            command = self._get_prefix_int(size=1)
            match command:
                case Command.heartbeat.value:
                    print("Heartbeat not implemented.")
                    break
                case Command.start.value:
                    self.delegate.start_response()
                case Command.stop.value:
                    self.delegate.stop_response()
                case Command.data.value:
                    size = self._get_prefix_int(size=4)
                    data = self._get_data(size)
                    self.delegate.data_response(data)
                case Command.data.enableVideoModule.value:
                    self.delegate.enable_video_recording_module()
                case _:
                    print(
                        "Network Protocol Violation: Unknown Command received '{command}'"
                    )

    def _get_prefix_int(self, size: int) -> int:
        prefix = self._connection.receive(size)
        prefix = int.from_bytes(prefix, "big", signed=False)
        return prefix

    def _get_data(self, size: int) -> bytes:
        data = self._connection.receive(size)
        return data

    def close(self) -> None:
        self._active = False
        # TODO: Implement close function
        print("Not fully implemented")


class MainControllerSensApp(NetworkControllerDelegate):

    def __init__(self, service_name: str) -> None:
        self._network_controller: NetworkController = None
        self._video_capture = VideoCapture(fps=30, capture_device_id=0)
        self._network_browser = NetworkBrowser(
            service_name, self._connection_found_handler
        )
        self.relative_recordings_folder_path = "./recordings"
        self.common_recording_session_name = None
        self.recording_start_time = None

    def _connection_found_handler(self, connection: Connection):
        print("Connection Found")
        threading.Thread(target=self._network_browser.close).start()
        self._network_controller = NetworkController(connection, self)
        self._network_controller.start()
        # self._video_capture.start_preview()

    def start_response(self) -> None:
        print("Start received")

        self.recording_start_time = datetime.now()
        self.common_recording_session_name = self.recording_start_time.strftime(
            "%d-%m-%Y_%H-%M-%S"
        )

        self.base_session_folder_path = f"{self.relative_recordings_folder_path}/{self.common_recording_session_name}"
        self.session_folder_path = f"{self.base_session_folder_path}/{self.common_recording_session_name}"
        try:
            os.makedirs(self.session_folder_path)
        except FileExistsError:
            self.close()
            print("Error: Current Recordings Folder already exists")
            exit()

        self._video_capture.start(
            self.session_folder_path, self.common_recording_session_name
        )

    def stop_response(self) -> None:
        print("Stop received")
        print("Stopping Video Capture")
        self._video_capture.stop()
        print("Video Capture Stopped")
        # self._video_capture.start_preview()

    def data_response(self, data: object) -> None:
        print("Received", len(data), "bytes of Motion Data")

        # Inserting Video Start time and app name into data (json).
        start_pos = data.find(b"{")
        formatted_start_time = self.recording_start_time.strftime(
            "%H:%M:%S.%f"
        )
        json_attributes = '"applicationName":"sensApp",'
        json_attributes += '"videoStartTime":"' + formatted_start_time + '",'
        json_attributes = bytes(json_attributes, "utf-8")
        data = data[:start_pos+1] + json_attributes + data[start_pos+1:]

        # Writing File.
        file_path = f"{self.session_folder_path}/{self.common_recording_session_name}.json"
        if os.path.isfile(file_path):
            print("Warning: .json recording file already exists. Appending Data...")
        with open(file_path, "ab") as f:
            f.write(data)
        print("Finished writing Motion Data")
        print("Zipping Recording Folder")
        subprocess.run([
            "./zipAndRemove",
            self.base_session_folder_path,
            self.common_recording_session_name
        ])

    def close(self) -> None:
        self._video_capture.close()
        self._network_browser.close()
        if self._network_controller is not None:
            self._network_controller.close()


class MainControllerConsistencyTracker(NetworkControllerDelegate):

    def __init__(self, service_name: str) -> None:
        self._network_controller: NetworkController = None
        self._video_capture = None
        self._network_browser = NetworkBrowser(
            service_name, self._connection_found_handler
        )
        self.relative_recordings_folder_path = "./recordingsConsistencyTracker"
        self.common_recording_session_name = None
        self.recording_start_time = None

        self.file_names_in_order = ["rawMotionData",  # "filteredMotionData", # When Debugging file is send as well
                                    "rotRateXAnnotations", "rotRateYAnnotations", "rotRateZAnnotations",
                                    "gravityXAnnotations", "gravityYAnnotations", "gravityZAnnotations",
                                    "linAccXAnnotations", "linAccYAnnotations", "linAccZAnnotations"]
        self.files_received = 0
        self.expecting_data = False

    def _connection_found_handler(self, connection: Connection):
        print("Connection Found")
        threading.Thread(target=self._network_browser.close).start()
        self._network_controller = NetworkController(connection, self)
        self._network_controller.start()
        # self._video_capture.start_preview()

    def _create_recording_folder(self) -> None:
        self.recording_start_time = datetime.now()
        self.common_recording_session_name = self.recording_start_time.strftime(
            "%d-%m-%Y_%H-%M-%S"
        )

        self.base_session_folder_path = f"{self.relative_recordings_folder_path}/{self.common_recording_session_name}"
        self.session_folder_path = f"{self.base_session_folder_path}/{self.common_recording_session_name}"
        try:
            os.makedirs(self.session_folder_path)
        except FileExistsError:
            self.close()
            print("Error: Current Recordings Folder already exists")
            exit()

    def enable_video_recording_module(self) -> None:
        print("Enable Video Module Received")
        if self._video_capture is not None:
            print("Video Module already enabled")
            return
        self._video_capture = VideoCapture(fps=30, capture_device_id=0)

    def start_response(self) -> None:
        print("Start received")
        if self._video_capture is None:
            print("Video module not started, rejecting start")
            return

        self.expecting_data = False

        self._create_recording_folder()

        self._video_capture.start(
            self.session_folder_path, self.common_recording_session_name
        )

    def stop_response(self) -> None:
        print("Stop received")
        self.files_received = 0
        self.expecting_data = True

        if self._video_capture is not None and self._video_capture._is_recording:
            print("Stopping Video Capture")
            self._video_capture.stop()
            print("Video Capture Stopped")
        else:
            self._create_recording_folder()

    def data_response(self, data: object) -> None:
        print("Received", len(data), "bytes of Motion Data")
        if not self.expecting_data:
            print("Warning: But have not expected data. Ignoring")
            return

        if self.files_received == 1:  # < 2 when debug file is send as well.
            # Inserting Video Start time and app name into data (json).
            start_pos = data.find(b"{")
            formatted_start_time = self.recording_start_time.strftime(
                "%H:%M:%S.%f"
            )
            json_attributes = '"applicationName":"sensApp",'
            json_attributes += '"videoStartTime":"' + formatted_start_time + '",'
            json_attributes = bytes(json_attributes, "utf-8")
            data = data[:start_pos+1] + json_attributes + data[start_pos+1:]
            file_path = f"{self.session_folder_path}/{self.file_names_in_order[self.files_received]}.json"
        else:
            file_path = f"{self.base_session_folder_path}/{self.file_names_in_order[self.files_received]}.json"
        if os.path.isfile(file_path):
            print(f"Warning: {self.file_names_in_order[self.files_received]} .json file already exists. Appending Data...")
        with open(file_path, "ab") as f:
            f.write(data)
        self.files_received += 1

        if self.files_received >= len(self.file_names_in_order):
            self.expecting_data = False
            subprocess.run([
                "./zipAndRemove",
                self.base_session_folder_path,
                self.common_recording_session_name
            ])

    def close(self) -> None:
        # self._video_capture.close()
        self._network_browser.close()
        if self._network_controller is not None:
            self._network_controller.close()
