import cv2
import dlib
import numpy as np
import pyttsx3
import time
import json
import csv
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURATION ---
USER_MODE = "user"  # "user" or "doctor"
CALIBRATION_FILE = "calibration_profile.json"
LOG_FILE = "session_log.csv"
GRAPH_FILE = "pupil_graph.png"
SESSION_DURATION = 10 * 60  # 10 minutes

# --- INITIALIZATION ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_region(landmarks, eye_points):
    return np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points], np.int32)

def crop_eye(frame, eye_region):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(eye_region)
    cropped = eye[y:y+h, x:x+w]
    return cropped

def detect_pupil_size(eye_img):
    if eye_img.size == 0:
        return 0
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pupil = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(pupil)
        return radius
    return 0

def save_calibration(calibration_data):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration_data, f)

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return None

def multi_stage_calibration(cap, detector, predictor):
    directions = [
        ("straight", "Please look straight at the camera."),
        ("left", "Please look to your left."),
        ("right", "Please look to your right."),
        ("up", "Please look up."),
        ("down", "Please look down.")
    ]
    calibration_data = {}
    speak("Calibration will begin. Please follow the instructions.")
    for direction, prompt in directions:
        speak(prompt)
        pupil_sizes_left = []
        pupil_sizes_right = []
        while len(pupil_sizes_left) < 30:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            cv2.putText(frame, f"Calibrating: {direction}", (60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            progress = int((len(pupil_sizes_left) / 30) * 500)
            cv2.rectangle(frame, (60, 70), (60 + progress, 100), (0, 255, 0), -1)
            cv2.rectangle(frame, (60, 70), (560, 100), (255, 255, 255), 2)
            cv2.imshow("Calibration", frame)
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                left_eye_points = [36, 37, 38, 39, 40, 41]
                right_eye_points = [42, 43, 44, 45, 46, 47]
                left_eye_region = get_eye_region(landmarks, left_eye_points)
                right_eye_region = get_eye_region(landmarks, right_eye_points)
                left_eye = crop_eye(frame, left_eye_region)
                right_eye = crop_eye(frame, right_eye_region)
                left_pupil = detect_pupil_size(left_eye)
                right_pupil = detect_pupil_size(right_eye)
                # Only accept reasonable pupil sizes
                if 1 < left_pupil < 20 and 1 < right_pupil < 20:
                    pupil_sizes_left.append(left_pupil)
                    pupil_sizes_right.append(right_pupil)
            else:
                cv2.putText(frame, "Face not detected. Please align your face.", (60, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if cv2.waitKey(1) == 27:
                break
        calibration_data[direction] = {
            "left": float(np.mean(pupil_sizes_left)) if pupil_sizes_left else 0,
            "right": float(np.mean(pupil_sizes_right)) if pupil_sizes_right else 0
        }
    cv2.destroyWindow("Calibration")
    speak("Calibration complete.")
    save_calibration(calibration_data)
    return calibration_data

def plot_and_save_graph(log_file, graph_file):
    df = pd.read_csv(log_file)
    df['time_seconds'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'][0])).dt.total_seconds()
    plt.figure(figsize=(12, 4))
    plt.plot(df["time_seconds"], df["left_pupil"], label="Left Pupil")
    plt.plot(df["time_seconds"], df["right_pupil"], label="Right Pupil")
    seizure_times = df[df["event"] == "SEIZURE"]["time_seconds"]
    for t in seizure_times:
        plt.axvline(t, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pupil Size")
    plt.title("Pupil Size Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_file)
    plt.close()

# --- MAIN APP ---
cap = cv2.VideoCapture(0)

# Calibration: load or run
calibration_data = load_calibration()
if not calibration_data:
    calibration_data = multi_stage_calibration(cap, detector, predictor)
baseline_left = calibration_data["straight"]["left"]
baseline_right = calibration_data["straight"]["right"]

prev_left, prev_right = baseline_left, baseline_right
threshold_change = 5
face_not_found_counter = 0
warning_played = False
paused = False

# Session logging
log_file = open(LOG_FILE, "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "left_pupil", "right_pupil", "event"])

speak("Session started. Press P to pause or resume, C to cancel, or Escape to exit.")

session_start = time.time()
start_time = time.time()
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        cv2.putText(frame, "Align your face and look straight at the camera", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elapsed = int(time.time() - session_start)
        remaining = max(0, SESSION_DURATION - elapsed)
        mins, secs = divmod(remaining, 60)
        timer_text = f"Time left: {mins:02d}:{secs:02d}"
        cv2.putText(frame, timer_text, (60, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if len(faces) == 0:
            face_not_found_counter += 1
            if face_not_found_counter > 20:
                cv2.rectangle(frame, (60, 420), (700, 470), (0, 0, 0), -1)
                cv2.putText(frame, "Please look into the camera!", (80, 455),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                if not warning_played:
                    speak("Please look into the camera.")
                    warning_played = True
            event = "NO_FACE"
            left_pupil = right_pupil = 0
        else:
            face_not_found_counter = 0
            warning_played = False
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye_points = [36, 37, 38, 39, 40, 41]
                right_eye_points = [42, 43, 44, 45, 46, 47]
                left_eye_region = get_eye_region(landmarks, left_eye_points)
                right_eye_region = get_eye_region(landmarks, right_eye_points)
                left_eye = crop_eye(frame, left_eye_region)
                right_eye = crop_eye(frame, right_eye_region)
                left_eye_zoom = cv2.resize(left_eye, (200, 100))
                right_eye_zoom = cv2.resize(right_eye, (200, 100))
                left_pupil = detect_pupil_size(left_eye)
                right_pupil = detect_pupil_size(right_eye)

                sudden_change_left = (
                    abs(left_pupil - baseline_left) > threshold_change and
                    abs(left_pupil - prev_left) > threshold_change
                )
                sudden_change_right = (
                    abs(right_pupil - baseline_right) > threshold_change and
                    abs(right_pupil - prev_right) > threshold_change
                )
                asymmetry = abs(left_pupil - right_pupil) > 6

                event = ""
                if sudden_change_left or sudden_change_right or asymmetry:
                    cv2.rectangle(frame, (80, 420), (600, 470), (0, 0, 255), -1)
                    cv2.putText(frame, "⚠️ Possible Seizure Detected", (100, 455),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    if not warning_played:
                        speak("Possible seizure detected. Please seek help.")
                        warning_played = True
                    event = "SEIZURE"
                else:
                    warning_played = False

                prev_left = left_pupil
                prev_right = right_pupil

                # Show zoomed eyes and pupil sizes
                eye_view = np.hstack((left_eye_zoom, right_eye_zoom))
                cv2.putText(eye_view, f"Left: {left_pupil:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(eye_view, f"Right: {right_pupil:.2f}", (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Zoomed Eyes View", eye_view)
                cv2.moveWindow("Zoomed Eyes View", 100, 100)

                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for (x, y) in left_eye_region:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                for (x, y) in right_eye_region:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Log data
        timestamp = datetime.datetime.now().isoformat()
        csv_writer.writerow([timestamp, left_pupil, right_pupil, event])
        log_file.flush()

        cv2.imshow("Live Feed", frame)

        # --- Fixed time session end ---
        if elapsed >= SESSION_DURATION:
            speak("Session time is over. Saving data.")
            break

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('p'):
        paused = not paused
        if paused:
            speak("Paused. Press P to resume.")
        else:
            speak("Resumed.")
    elif key == ord('c'):
        speak("Session cancelled.")
        break

log_file.close()
plot_and_save_graph(LOG_FILE, GRAPH_FILE)
speak("Session ended. Summary and graph saved.")

# Print summary for doctor mode
if USER_MODE == "doctor":
    df = pd.read_csv(LOG_FILE)
    print("Session Summary:")
    print(df["event"].value_counts())
    print(f"Average left pupil: {df['left_pupil'].mean():.2f}")
    print(f"Average right pupil: {df['right_pupil'].mean():.2f}")

cap.release()
cv2.destroyAllWindows()
