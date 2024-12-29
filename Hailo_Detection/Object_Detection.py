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
import time

class user_app_callback_class(app_callback_class):
    def __init__(self, shared_dict):
        super().__init__()
        self.shared_dict = shared_dict
        self.frame = None

    def set_person_count(self, count):
        self.shared_dict["person_count"] = count

    def get_person_count(self):
        return self.shared_dict.get("person_count", 0)

    def set_frame(self, frame):
        self.frame = frame

    def save_frame(self):
        if self.frame is not None:
            original_height, original_width = self.frame.shape[:2]

            target_width = 2560
            target_height = 1440

            original_aspect_ratio = original_width / original_height
            target_aspect_ratio = target_width / target_height

            if original_aspect_ratio > target_aspect_ratio:
                new_width = int(original_height * target_aspect_ratio)
                x_offset = (original_width - new_width) // 2
                cropped_frame = self.frame[:, x_offset:x_offset + new_width]
                resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
            else:
                new_height = int(original_width / target_aspect_ratio)
                y_offset = (original_height - new_height) // 2
                cropped_frame = self.frame[y_offset:y_offset + new_height, :]
                resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

            cv2.imwrite("/home/minjun/Desktop/image.jpg", resized_frame)
            print("Saved image to /home/minjun/Desktop/image.jpg")

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

    frame = get_numpy_from_buffer(buffer, format, width, height)
    
    if format == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

def capture_image_periodically(user_data):
    user_data.save_frame()
    return True

def run_detection(shared_dict):
    Gst.init(None)
    user_data = user_app_callback_class(shared_dict)
    
    GLib.timeout_add_seconds(3, capture_image_periodically, user_data)
    
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
