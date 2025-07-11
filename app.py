from flask import Flask, render_template, request, Response, redirect
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO
weights_path = 'yolo/yolov4.weights'
config_path = 'yolo/yolov4.cfg'
labels_path = 'yolo/coco.names'  # Correct path for coco.names

# Load class labels
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects_yolo(image):
    # Check if the input is a file path or a frame (for webcam)
    if isinstance(image, str):  # If it's a path, load the image
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Failed to load image from the path: {image}")
    
    height, width = image.shape[:2]

    # Preprocessing
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Detection logic...
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            result.append(label)

            # Draw the bounding box and label
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Encode the processed image for web display
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return len(result), result, encoded_image


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform object detection
            object_count, object_details, processed_image = detect_objects_yolo(file_path)

            return render_template('result.html', object_count=object_count, object_details=object_details, image=processed_image)

    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    cap = cv2.VideoCapture(0)  # Capture from webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect objects in the webcam frame
            object_count, object_details, processed_image = detect_objects_yolo(frame)
            
            # Convert processed frame back to bytes for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(debug=True)
