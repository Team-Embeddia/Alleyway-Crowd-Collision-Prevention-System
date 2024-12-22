import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from instance_segmentation_pipeline import GStreamerInstanceSegmentationApp
from multiprocessing import Manager

class user_app_callback_class(app_callback_class):
    def __init__(self, shared_dict):
        super().__init__()
        self.shared_dict = shared_dict

    def set_person_count(self, count):
        self.shared_dict["person_count"] = count

    def get_person_count(self):
        return self.shared_dict.get("person_count", 0)

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    person_count = 0
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        if label == "person":
            person_count += 1

    user_data.set_person_count(person_count)
    print(f"Frame count: {user_data.get_count()}, Person count: {person_count}")

    return Gst.PadProbeReturn.OK

def run_detection(shared_dict):
    Gst.init(None)
    user_data = user_app_callback_class(shared_dict)
    app = GStreamerInstanceSegmentationApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    from multiprocessing import Process, Manager
    from HTTP import run_http_server

    with Manager() as manager:
        shared_dict = manager.dict({"person_count": 0})

        detection_process = Process(target=run_detection, args=(shared_dict,))
        http_process = Process(target=run_http_server, args=(shared_dict,))

        detection_process.start()
        http_process.start()

        detection_process.join()
        http_process.join()
