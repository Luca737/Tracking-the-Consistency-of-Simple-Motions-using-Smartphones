"""Author: Nicola Vidovic"""

import threading
import time
from typing import Callable

import cv2


class VideoCapture():
    _min_fps = 1
    _max_fps = 60

    def __init__(self, fps: float, capture_device_id: int) -> None:
        self._fps = max(min(fps, VideoCapture._max_fps), VideoCapture._min_fps)
        self._fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self._cap = cv2.VideoCapture(capture_device_id)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._dimensions = (width, height)
        self._recording_thread = None
        self._preview_thread = None
        self._video_writer = RealTimeVideoWriter()

        self._run_preview = False
        self._is_recording = False
        self._stop_recording = False
        self._stop_func_lock = threading.Lock()

    def start(self, out_folder_path: str, out_file_name: str) -> bool:
        if self._is_recording:
            print("Already recording")
            return False

        # Video format must always be .mp4
        if "." not in out_file_name:
            out_file_name += ".mp4"
        elif not out_file_name.lower().endswith(".mp4"):
            print("Output file must be of type '.mp4'")
            return False

        self._out = cv2.VideoWriter(
            f"{out_folder_path}/{out_file_name}", self._fourcc, self._fps, self._dimensions
        )

        # Start recording in a separate Thread
        self._recording_thread = threading.Thread(
            target=self._start_record_loop
        )
        self._recording_thread.start()

        return True

    def start_preview(self) -> bool:
        # TODO: Opening windows and displaying frames does not work on MACOS: In separate Thread - Illegal Hardware Instruction; In main Thread does nothing.
        if self._preview_thread is not None and self._preview_thread.is_alive():
            print("Preview already active")
            return False
        self._preview_thread = threading.Thread(target=self._start_preview)
        self._preview_thread.start()
        return True

    def stop_preview(self) -> None:
        if not self._run_preview:
            print("No preview to stop")
            return
        self._run_preview = False

    def _start_preview(self) -> None:
        fps = 5
        pause = 1/fps
        window_name = "preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._run_preview = True
        while self._run_preview:
            ret, frame = self._cap.read()
            if not ret:
                print("Preview stopped unexpectedly")
                break
            cv2.imshow(window_name, frame)
            time.sleep(pause)
        cv2.destroyWindow(window_name)

    def _start_record_loop(self) -> None:
        self._run_preview = False
        self._is_recording = True
        self._stop_recording = False
        # --- Measuring fps ---
        start_time = time.time()
        last_time = start_time
        self._video_writer.setup(self._out, self._fps, start_time)
        frames = 0
        # ---------------------
        print("Actual start time of recording:", start_time)
        while not self._stop_recording:
            ret, frame = self._cap.read()
            frame_time = time.time()
            if not ret:
                threading.Thread(target=self.stop).start()
                print("Recording stopped unexpectedly")
                break
            self._video_writer.write(frame, frame_time)
            frames += 1
            if frames % 10 == 0:
                print("fps:", 10 / (frame_time - last_time), end="\r")
                last_time = frame_time
        end_time = time.time()
        self._video_writer.finnish(end_time)

        print("Total fps:", frames / (end_time - start_time))
        print("Video recording length:", end_time - start_time)

    def stop(self) -> None:
        if not self._stop_func_lock.acquire(blocking=False):
            return

        if self._is_recording:
            self._stop_recording = True
            self._recording_thread.join()
            self._out.release()
            self._is_recording = False
        else:
            print("No recording to stop")

        self._stop_func_lock.release()

    def close(self) -> None:
        if self._is_recording:
            self.stop()
        self._cap.release()

    def __del__(self):
        self.close()


class RealTimeVideoWriter():

    def __init__(self) -> None:
        # User set constants
        self._video_start_time = None
        self._video_writer = None

        # Frame writer method used by User
        self.write = None

        # Class managed temp data
        self._last_frame = None
        self._frame_count = None
        self._bias = None  # How much earlier a frame is allowed to come.
        self._spf = None   # Seconds per frame.

    def setup(self, video_writer: Callable, fps: float, video_start_time: float):
        self._video_start_time = video_start_time
        self._video_writer = video_writer

        self._spf = 1/fps
        self._bias = self._spf * 0.1
        self._last_frame = None
        self._frame_count = 0

        self.write = self._write_first_frame

    def _write_first_frame(self, frame, frame_time: float):
        self._last_frame = frame
        self.write = self._write_frame
        self.write(frame, frame_time)

    def _write_frame(self, frame, frame_time: float):
        # Calculate how many frames have to be drawn.
        current_time = frame_time - self._video_start_time
        supposed_frames_since_start = int(
            (current_time - self._bias) // self._spf
        ) + 1  # First frame is at time zero
        num_frames_to_draw = supposed_frames_since_start - self._frame_count

        for _ in range(num_frames_to_draw):
            self._video_writer.write(self._last_frame)

        self._last_frame = frame
        self._frame_count += num_frames_to_draw

    def finnish(self, end_time: float):
        if self._last_frame is None:
            return
        self._write_frame(None, end_time)


if __name__ == "__main__":
    vid_cap = VideoCapture(30, 0)
    vid_cap._start_preview()
