from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
import base64

app = Flask(__name__)

# Path to shape predictor file
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor
predictor = dlib.shape_predictor(predictor_path)

def detect_gaze(landmarks, frame, gray):
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                 (landmarks.part(43).x, landmarks.part(43).y),
                                 (landmarks.part(44).x, landmarks.part(44).y),
                                 (landmarks.part(45).x, landmarks.part(45).y),
                                 (landmarks.part(46).x, landmarks.part(46).y),
                                 (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    cv2.polylines(mask, [right_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [right_eye_region], 255)
    right_eye = cv2.bitwise_and(gray, gray, mask=mask)

    left_eye_center = (landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2
    right_eye_center = (landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2

    if left_eye_center[0] < width // 2:
        return "Focused"
    else:
        return "Not Focused"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        print("Received image for processing.")
        data = request.json
        image_data = data['image'].split(",")[1]
        image_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        result = {"status": "error", "message": "No face detected"}
        
        for face in faces:
            landmarks = predictor(gray, face)
            gaze = detect_gaze(landmarks, img, gray)
            result = {"status": "success", "message": gaze}

        return jsonify(result)
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000, ssl_context=('server.crt', 'server.key'))
