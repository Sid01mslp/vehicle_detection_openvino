import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.runtime import Core

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture

from openvino.model_zoo.model_api.models import OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD', 'HETERO', 'HDDL']

# General variables list #-----------------------------------------------------------------------------------------#
# Help: An input to process. The input must be a single image,a folder of images, video file or camera id.
input = 'head-pose-face-detection-male.mp4'

# Help: optional. Enable reading the input in a loop
loop = False

# Help: Optional. Name of the output file(s) to save.
output = ''

# Help: Optional. Number of frames to store in output.If 0 is set, all frames are stored.
output_limit = 1000

# Help: Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720.
# Input frame size used by default.
output_resolution = None

# Help: Optional. Don't show output.
no_show = None

# Help: Optional. Crop the input stream to this resolution.
crop_size = (0, 0)

# Help: Optional. Algorithm for face matching. Default: HUNGARIAN.
match_algo = 'HUNGARIAN'

# Help: Optional. List of monitors to show initially.
utilization_monitors = ''

# Faces database variables list #----------------------------------------------------------------------------------#
# Help: Optional. Path to the face images directory.
fg = ''

# Help: Optional. Use Face Detection model to find faces on the face images, otherwise use full-images.
run_detector = None

# Help: Optional. Allow to grow faces gallery and to dump on disk. Available only if --no_show option is off.
allow_grow = None

# Models variables list #------------------------------------------------------------------------------------------#
# Help: Required. Path to an .xml file with Face Detection model.
m_fd = 'intel/models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml'

# Help: Required. Path to an .xml file with Facial Landmarks Detection model.
m_lm = 'intel/models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'

# Help: Required. Path to an .xml file with Face Reidentification model.
m_reid = 'intel/models/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml'

# Help: Optional. Specify the input size of detection model for reshaping. Example: 500 700.
fd_input_size = (0, 0)

# Inference options variables list #-------------------------------------------------------------------------------#
# Help: Optional. Target device for Face Detection model. Default value is CPU.
d_fd = 'CPU'

# Help: Optional. Target device for Facial Landmarks Detection model. Default value is CPU.
d_lm = 'CPU'

# Help: Optional. Target device for Face Reidentification model. Default value is CPU.
d_reid = 'CPU'

# Help: Optional. Be more verbose.
verbose = None

# Help: Optional. Probability threshold for face detections.
t_fd = 0.6

# Help: Optional. Cosine distance threshold between two vectors for face identification.
t_id = 0.3

# Help: Optional. Scaling ratio for bboxes passed to face recognition.
exp_r_fd = 1.15


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self):
        self.allow_grow = allow_grow and not no_show

        log.info('OpenVINO Runtime')
        core = Core()

        self.face_detector = FaceDetector(core, m_fd,
                                          fd_input_size,
                                          confidence_threshold=t_fd,
                                          roi_scale_factor=exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, m_lm)
        self.face_identifier = FaceIdentifier(core, m_reid,
                                              match_threshold=t_id,
                                              match_algo=match_algo)

        self.face_detector.deploy(d_fd)
        self.landmarks_detector.deploy(d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(fg))
        self.faces_database = FacesDatabase(fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if run_detector else None, no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                        (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                        (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]


def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame


def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2: (fh + crop_size[1]) // 2,
           (fw - crop_size[0]) // 2: (fw + crop_size[0]) // 2, :]


def main():
    global output_resolution
    cap = open_images_capture(input, loop)

    frame_processor = FrameProcessor()

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None

    if crop_size[0] > 0 and crop_size[1] > 0:
        input_crop = np.array(crop_size)
    elif not (crop_size[0] == 0 and crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')

    video_writer = cv2.VideoWriter()

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        if input_crop:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], output_resolution)
            if output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if output and not video_writer.open(output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)

        presenter.drawGraphs(frame)

        frame = draw_detections(frame, frame_processor, detections, output_transform)

        metrics.update(start_time, frame)

        frame_num += 1
        if video_writer.isOpened() and (output_limit <= 0 or frame_num <= output_limit):
            video_writer.write(frame)

        if not no_show:
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)


    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


# if __name__ == '__main__':
#     main()



# Flask part
from flask import Flask, render_template, Response

app = Flask(__name__)

cap = open_images_capture(input, loop)
frame_processor = FrameProcessor()



output_transform = None
vidcap = cv2.VideoCapture('head-pose-face-detection-male.mp4')
success, frame = vidcap.read()


def generate_frames():

    output_resolution = None
    frame_num = 0
    while True:
        # frame = cap.read()
        success, frame = vidcap.read()

        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], output_resolution)
            if output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])

        detections = frame_processor.process(frame)

        frame = draw_detections(frame, frame_processor, detections, output_transform)

        frame_num += 1

        print(frame)

        if not no_show:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # print("-----------")
        # print(success)
        # print("-----------")
        #
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + xxxxxxx + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)


