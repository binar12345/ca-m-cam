import cv2
import numpy as np
import pygame
from flask import Flask, render_template, Response
import math
import threading
import mediapipe as mp
import time

# Initialize pygame for sound
pygame.mixer.init()
beep_sound = pygame.mixer.Sound('static/beep.wav')  # Load the beep sound
beep_sound.set_volume(1.0)  # Set the beep sound to maximum volume

app = Flask(__name__)

# Initialize MediaPipe Holistic model (for full-body tracking)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = None

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    # Set camera resolution (adjust the values as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None

# Frame size (ensure consistent size across the application)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Camera calibration: Focal length (calibrated value should be used)
KNOWN_FACE_WIDTH = 15  # cm (average face width)
FOCAL_LENGTH = 600  # Calibrate this based on the camera and environment
FAR_DISTANCE = 1000  # cm (10 meters)

def calculate_distance_between_points(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def generate_frames():
    global cap
    # Load the pre-trained model for age estimation (using Caffe framework)
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

    # Age list that corresponds to the model's output
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_frame_time = time.time()
        while True:
            # Check if we need to start the camera
            if cap is None or not cap.isOpened():
                start_camera()

            ret, frame = cap.read()
            if not ret:
                break

            # Limit the frame rate to around 30 fps to reduce lag
            current_time = time.time()
            elapsed_time = current_time - last_frame_time
            if elapsed_time < 1.0 / 30:  # 30 frames per second
                continue
            last_frame_time = current_time

            # Resize the frame to ensure consistent resolution
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process full-body tracking
            results = holistic.process(rgb_frame)

            # Draw full-body pose, face, and hands
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Detect faces using OpenCV's pre-trained Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Crop the face region
                face = frame[y:y + h, x:x + w]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                # Pass the blob through the network and get the age prediction
                age_net.setInput(blob)
                age_preds = age_net.forward()

                # Get the index of the predicted age
                age = age_preds[0].argmax()
                age_text = age_list[age]  # Convert index to actual age range

                # Draw a bounding box and age prediction on the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Age: {age_text}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check distance between left hand and left shoulder
            if results.left_hand_landmarks and results.pose_landmarks:
                left_wrist = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]

                left_wrist_position = (left_wrist.x * FRAME_WIDTH, left_wrist.y * FRAME_HEIGHT)
                left_shoulder_position = (left_shoulder.x * FRAME_WIDTH, left_shoulder.y * FRAME_HEIGHT)

                distance_left_hand_body = calculate_distance_between_points(left_wrist_position, left_shoulder_position)

                # If the left hand is within 100 cm of the body, play beep
                if distance_left_hand_body < 100:  # 1 meter = 100 cm
                    beep_sound.play()

            # Check distance between right hand and right shoulder
            if results.right_hand_landmarks and results.pose_landmarks:
                right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

                right_wrist_position = (right_wrist.x * FRAME_WIDTH, right_wrist.y * FRAME_HEIGHT)
                right_shoulder_position = (right_shoulder.x * FRAME_WIDTH, right_shoulder.y * FRAME_HEIGHT)

                distance_right_hand_body = calculate_distance_between_points(right_wrist_position, right_shoulder_position)

                # If the right hand is within 100 cm of the body, play beep
                if distance_right_hand_body < 100:  # 1 meter = 100 cm
                    beep_sound.play()

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    try:
        # Start the camera in a separate thread to prevent blocking the app
        camera_thread = threading.Thread(target=start_camera)
        camera_thread.start()

        app.run(debug=True, use_reloader=False)  # Disable the reloader to avoid issues with multi-threaded video feed
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_camera()  # Ensure the camera is released when the app stops
