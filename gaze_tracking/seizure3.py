import cv2
import dlib
import numpy as np
import pyttsx3
import datetime
import os
import csv
import json

CALIBRATION_FILE = "calibration_profile.json"
LOG_FILE = "session_log.csv"
FRAME_FILE = "frame.jpg"
EYE_FILE = "eyes.jpg"
SESSION_DURATION = 10 * 60
VOICE_ENABLED = True
THRESHOLD_CHANGE = 5

engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    if VOICE_ENABLED:
        engine.say(text)
        engine.runAndWait()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_region(landmarks, points):
    return np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in points], np.int32)

def crop_eye(frame, region):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(region)
    return eye[y:y+h, x:x+w]

def detect_pupil_size(eye_img):
    if eye_img.size == 0:
        return 0
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pupil = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(pupil)
        return radius
    return 0

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return {
        "straight": {"left": 10.0, "right": 10.0}
    }

cap = cv2.VideoCapture(0)
calib = load_calibration()
baseline_left = calib["straight"]["left"]
baseline_right = calib["straight"]["right"]
prev_left, prev_right = baseline_left, baseline_right

with open(LOG_FILE, "w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "left_pupil", "right_pupil", "event"])

    speak("Seizure monitoring started.")
    start_time = datetime.datetime.now()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        left_pupil, right_pupil, event = 0, 0, ""

        if faces:
            face = faces[0]
            landmarks = predictor(gray, face)

            left_pts = [36, 37, 38, 39, 40, 41]
            right_pts = [42, 43, 44, 45, 46, 47]
            left_eye_reg = get_eye_region(landmarks, left_pts)
            right_eye_reg = get_eye_region(landmarks, right_pts)
            left_eye = crop_eye(frame, left_eye_reg)
            right_eye = crop_eye(frame, right_eye_reg)

            left_pupil = detect_pupil_size(left_eye)
            right_pupil = detect_pupil_size(right_eye)

            sudden_change_left = abs(left_pupil - baseline_left) > THRESHOLD_CHANGE and abs(left_pupil - prev_left) > THRESHOLD_CHANGE
            sudden_change_right = abs(right_pupil - baseline_right) > THRESHOLD_CHANGE and abs(right_pupil - prev_right) > THRESHOLD_CHANGE
            asymmetry = abs(left_pupil - right_pupil) > 6

            if sudden_change_left or sudden_change_right or asymmetry:
                event = "SEIZURE"
                cv2.putText(frame, "⚠️ Seizure Detected", (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                speak("Warning. Possible seizure detected.")

            prev_left = left_pupil
            prev_right = right_pupil

            left_eye_zoom = cv2.resize(left_eye, (200, 100))
            right_eye_zoom = cv2.resize(right_eye, (200, 100))
            eye_view = np.hstack((left_eye_zoom, right_eye_zoom))
            cv2.imwrite(EYE_FILE, eye_view)

        else:
            event = "NO_FACE"
            cv2.putText(frame, "Align your face", (100, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        timestamp = datetime.datetime.now().isoformat()
        writer.writerow([timestamp, left_pupil, right_pupil, event])
        log_file.flush()

        cv2.imwrite(FRAME_FILE, frame)

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if (datetime.datetime.now() - start_time).total_seconds() >= SESSION_DURATION:
            break

speak("Session complete. Data saved.")
cap.release()
cv2.destroyAllWindows()
