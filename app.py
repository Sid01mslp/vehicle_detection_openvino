from flask import Flask, render_template, redirect, request, send_from_directory, Response
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
import cv2

from face_recognition.frame_processor import FrameProcessor
from face_recognition.draw_detections import draw_detections

from openvino.model_zoo.model_api.models import OutputTransform


app = Flask(__name__)
app.config['UPLOAD_DIRECTORY'] = 'static/uploads/'
app.config['THUMBNAILS_DIRECTORY'] = 'static/thumbnails/'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = ['.mp4']

input_video_file_name = ''


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_DIRECTORY'])
    images = []
    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images.append(file)

    return render_template('index.html', images=images)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file:
            extension = os.path.splitext(file.filename)[1].lower()
            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'Selected file is not a video.'
            file.save(os.path.join(
                app.config['UPLOAD_DIRECTORY'],
                secure_filename(file.filename)
            ))
    except RequestEntityTooLarge:
        return 'File is larger than the 200MB limit.'
    return redirect('/')


@app.route('/thumbnail/<filename>', methods=['GET'])
def thumbnail(filename):
    vcap = cv2.VideoCapture(os.path.join(
        app.config['UPLOAD_DIRECTORY'],
        secure_filename(filename)
    ))
    res, im_ar = vcap.read()
    while res:
        res, im_ar = vcap.read()
        if im_ar.mean() > 60:
            res, im_ar = vcap.read()
            break

    cv2.imwrite(os.path.join(
        app.config['THUMBNAILS_DIRECTORY'],
        secure_filename(os.path.splitext(filename)[0] + ".jpg")
    ), im_ar)
    return send_from_directory(
        app.config['THUMBNAILS_DIRECTORY'],
        os.path.splitext(filename)[0] + ".jpg"
    )


@app.route('/object-detection/<filename>', methods=['GET'])
def object_detection(filename):
    global input_video_file_name
    input_video_file_name = filename
    return render_template('video_player.html',)


def generate_frames(frame_processor, vidcap):
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

        if not False:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    frame_processor = FrameProcessor()

    video_cap = cv2.VideoCapture(os.path.join(
        app.config['UPLOAD_DIRECTORY'],
        secure_filename(input_video_file_name)
    ))

    return Response(
        generate_frames(
            frame_processor,
            video_cap
        ), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
