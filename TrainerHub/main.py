"""Author: Nicola Vidovic"""

import signal
import sys
import threading
from time import sleep

from controllers import MainControllerConsistencyTracker, MainControllerSensApp


def ctrl_c_handler(signum, frame):
    global app
    print("Shutting down...")
    app.close()
    sleep(2)
    exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, ctrl_c_handler)
    # app = MainControllerSensApp("_sensApp._tcp.local.")
    app = MainControllerConsistencyTracker("_ctApp._tcp.local.")
    print("Ready")

    while True:
        input()
