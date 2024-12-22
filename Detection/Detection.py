import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from instance_segmentation_pipeline import GStreamerInstanceSegmentationApp

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            if user_data.use_frame:
                masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
                if len(masks) != 0:
                    mask = masks[0]
                    mask_height = mask.get_height()
                    mask_width = mask.get_width()
                    data = np.array(mask.get_data())
                    data = data.reshape((mask_height, mask_width))
                    mask_width = mask_width * 4
                    mask_height = mask_height * 4
                    data = cv2.resize(data, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
                    string_to_print += f"Mask shape: {data.shape}, "
                    string_to_print += f"Base coordinates ({int(bbox.xmin() * width)},{int(bbox.ymin() * height)})\n"

    print(string_to_print)

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerInstanceSegmentationApp(app_callback, user_data)
    app.run()