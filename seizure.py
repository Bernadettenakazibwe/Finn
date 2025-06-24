# seizure.py
import cv2
import dlib
import numpy as np
import datetime
import csv
import time

LOG_FILE = "session_log.csv"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# --- Calibration Setup ---
def show_instruction(text):
    black = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(black, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Calibration", black)
    cv2.waitKey(3000)

# --- Detect pupil radius from cropped eye image ---
def detect_pupil_radius(eye_img):
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

# --- Extract eye image from frame ---
def get_eye_region(frame, eye_points, landmarks):
    points = [(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points]
    x, y, w, h = cv2.boundingRect(np.array(points))
    margin = 5
    # Ensure coordinates are within frame bounds
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, frame.shape[1])
    y2 = min(y + h + margin, frame.shape[0])
    return frame[y1:y2, x1:x2]

# --- Draw arrow on frame ---
def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    length = 100
    thickness = 5
    color = (0, 255, 0)

    if direction == "up":
        pt1 = center
        pt2 = (center[0], center[1] - length)
    elif direction == "down":
        pt1 = center
        pt2 = (center[0], center[1] + length)
    elif direction == "left":
        pt1 = center
        pt2 = (center[0] - length, center[1])
    elif direction == "right":
        pt1 = center
        pt2 = (center[0] + length, center[1])
    else:
        return None  # Invalid direction

    # Draw the arrow line
    cv2.arrowedLine(frame, pt1, pt2, color, thickness, tipLength=0.2)
    return frame

# --- Main Monitoring Logic ---
def monitor():
    cap = cv2.VideoCapture(0)
    
    # Calibration phase
    show_instruction("Look straight ahead")
    baseline = []
    
    for _ in range(50):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue  # Skip this iteration if frame is invalid

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            continue
        face = faces[0]
        landmarks = predictor(gray, face)

        left_eye_img = get_eye_region(frame, [36,37,38,39,40,41], landmarks)
        right_eye_img = get_eye_region(frame, [42,43,44,45,46,47], landmarks)

        left_radius = detect_pupil_radius(left_eye_img)
        right_radius = detect_pupil_radius(right_eye_img)
        if left_radius > 0 and right_radius > 0:
            baseline.append((left_radius + right_radius) / 2)
        time.sleep(0.05)

    if not baseline:
        print("Calibration failed. Try again.")
        cap.release()
        return

    baseline_value = np.mean(baseline)
    threshold = 5  # dynamic change from baseline
    print(f"Calibrated baseline pupil size: {baseline_value:.2f}")

    # Open CSV file
    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "left_pupil", "right_pupil", "event"])

        # Monitoring loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            event = "NO_FACE"
            left_radius = right_radius = 0

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)
                left_eye_img = get_eye_region(frame, [36,37,38,39,40,41], landmarks)
                right_eye_img = get_eye_region(frame, [42,43,44,45,46,47], landmarks)

                left_radius = detect_pupil_radius(left_eye_img)
                right_radius = detect_pupil_radius(right_eye_img)

                avg_radius = (left_radius + right_radius) / 2
                asymmetry = abs(left_radius - right_radius)

                if abs(avg_radius - baseline_value) > threshold or asymmetry > 4:
                    event = "SEIZURE"
                    cv2.putText(frame, "⚠ Seizure Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    event = "NORMAL"

            timestamp = datetime.datetime.now().isoformat()
            writer.writerow([timestamp, f"{left_radius:.2f}", f"{right_radius:.2f}", event])

            cv2.imshow("Live Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor()
