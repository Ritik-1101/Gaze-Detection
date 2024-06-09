# Flask Gaze Detection Application

## Description
This project is a Flask web application that uses a webcam to detect gaze using facial landmarks. The application uses OpenCV and Dlib for face detection and landmark estimation.

## Features
- Detects and displays facial landmarks on the webcam feed.
- Determines if the user is "Focused" or "Not Focused" based on eye position.

## Technologies Used
- Python
- Flask
- OpenCV
- Dlib
- NumPy

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- A webcam for video capture

### Installation

1. **Clone the Repository**
    ```sh
    git clone https://github.com/Ritik-1101/Gaze-Detection
    cd flask_gaze_app
    ```

2. **Create and Activate Virtual Environment**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate  # Linux/Mac
    ```

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

### Setup Dlib Shape Predictor
Download the shape predictor file from the following link and place it in the project directory:
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### Configuration
Ensure the path to the shape predictor file is correct in `app/routes.py`:
```python
predictor_path = '/shape_predictor_68_face_landmarks.dat'
