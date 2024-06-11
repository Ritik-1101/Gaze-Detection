from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data['image'].split(",")[1]
    image_data = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform gaze detection on the image
    # Replace the following line with your gaze detection code
    result = perform_gaze_detection(img)

    return jsonify(result=result)

def perform_gaze_detection(img):
    # Placeholder function for gaze detection
    # Implement your gaze detection algorithm here
    return {"status": "success", "message": "Gaze detected"}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
