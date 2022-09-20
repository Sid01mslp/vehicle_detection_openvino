#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
from pathlib import Path

import cv2
from visualizers import ColorPalette

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from openvino.model_zoo.model_api.models import DetectionModel, OutputTransform
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter
from vehicle_detection.draw_detection import draw_detections

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

# General Arguments
model = "intel/models/vehicle-detection-0202/FP16/vehicle-detection-0202.xml"
architecture_type = "ssd"
adapter = "openvino"  # choices=('openvino', 'ovms')
device = "CPU"

# Common model options Arguments
labels = None
prob_threshold = 0.5
resize_type = None  # choices=RESIZE_TYPES.keys()
input_size = (600, 600)
anchors = None
masks = None
layout = None
num_classes = None  # int

# Inference options Arguments
num_infer_requests = 0  # int
num_streams = ''
num_threads = None  # int

# Input/output options Arguments
loop = False
output = 'result.mp4'
output_limit = 1000
no_show = None
output_resolution = None  # (1280, 720)
utilization_monitors = ''

# Input transform options Arguments
reverse_input_channels = False
mean_values = None  # Example: 255.0 255.0 255.0
scale_values = None  # Example: 255.0 255.0 255.0

# Debug options Arguments
raw_output_message = False


# Execution process start
model_adapter = None

if adapter == 'openvino':
    plugin_config = get_user_config(device, num_streams, num_threads)
    model_adapter = OpenvinoAdapter(create_core(), model, device=device, plugin_config=plugin_config,
                                    max_num_requests=num_infer_requests, model_parameters={'input_layouts': layout})
elif adapter == 'ovms':
    model_adapter = OVMSAdapter(model)

configuration = {
    'resize_type': resize_type,
    'mean_values': mean_values,
    'scale_values': scale_values,
    'reverse_input_channels': reverse_input_channels,
    'path_to_labels': labels,
    'confidence_threshold': prob_threshold,
    'input_size': input_size,  # The CTPN specific
    'num_classes': num_classes,  # The NanoDet and NanoDetPlus specific
}

model = DetectionModel.create_model(architecture_type, model_adapter, configuration)

detector_pipeline = AsyncPipeline(model)


def detect_vehicle(video_captured_frame):
    global model, output_resolution

    palette = ColorPalette(len(model.labels) if model.labels else 100)

    next_frame_id = 0
    next_frame_id_to_show = 0
    output_transform = None

    while True:

        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]

        results = detector_pipeline.get_result(next_frame_id_to_show)

        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            frame = draw_detections(frame, objects, palette, model.labels, output_transform)

            if not no_show:
                return frame

            continue

        if detector_pipeline.is_ready():
            frame = video_captured_frame

            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break

            if next_frame_id == 0:
                output_transform = OutputTransform(frame.shape[:2], output_resolution)
                if output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])

            # Submit for inference
            detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame})
            next_frame_id += 1
        else:
            detector_pipeline.await_any()



