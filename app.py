from flask import Flask, render_template, request, redirect, jsonify
import os,cv2, whisper
import numpy as np
import re

model = whisper.load_model("base.en")
app = Flask(__name__)
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"
net = cv2.dnn.readNet(yolo_weights, yolo_config)

classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

def detect_objects(image_path,object_class):
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()

    detections = net.forward(layer_names)

    objects = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, width, height = list(map(int, obj[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])))
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                objects.append({
                    "label": classes[class_id],
                    "confidence": float(confidence),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                })
    filtered_objects = []
    bounding_boxes = []
    for obj in objects:
        if obj["label"].lower() in object_class:
            filtered_objects.append(obj)
            x, y, width, height = obj["x"], obj["y"], obj["width"], obj["height"]
            bounding_boxes.append((x, y, x + width, y + height))

    return filtered_objects, bounding_boxes

def draw_bounding_boxes(image_path, bounding_boxes):
    image = cv2.imread(image_path)

    for x1, y1, x2, y2 in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(image_path, image)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        image = request.files["image"]
        # object_class = request.form.get("object_class").lower()
        audio_file = request.files['audio_file']
        print(type(audio_file))
        # audio = request.files["audio"]
        path="./audios/sample.wav"
        audio_file.save(path)
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'})
        try:         
            response=model.transcribe(path,fp16=False) 
            text = response['text']
            re.sub('\W+','', text)
            object_class = text.lower()
        except Exception as e:
            return jsonify({'error-f': f"Error: {e}"})

        if image.filename == "":
            return redirect(request.url)

        if image:
            image_path = os.path.join("static", image.filename)
            image.save(image_path)

            objects, bounding_boxes = detect_objects(image_path, object_class)
            
            if bounding_boxes:
                draw_bounding_boxes(image_path, bounding_boxes)
            
            return render_template("index.html", image_path=image_path, objects=objects)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
