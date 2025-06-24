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
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import threading

# --- CONFIGURATION ---
USER_MODE = "user"  # "user" or "doctor"
CALIBRATION_FILE = "calibration_profile.json"
LOG_FILE = "session_log.csv"
GRAPH_FILE = "pupil_graph.png"
SESSION_DURATION = 10 * 60  # 10 minutes

# --- INITIALIZATION ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voice_enabled = True

def speak(text):
    if voice_enabled:
        engine.say(text)
        engine.runAndWait()

def speak_async(text):
    if voice_enabled:
        threading.Thread(target=speak, args=(text,), daemon=True).start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_region(frame, eye_points, landmarks):
    points = [(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points]
    x, y, w, h = cv2.boundingRect(np.array(points))
    margin = 5
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, frame.shape[1])
    y2 = min(y + h + margin, frame.shape[0])
    # Only return if region is valid
    if y2 > y1 and x2 > x1:
        return frame[y1:y2, x1:x2]
    else:
        return np.zeros((1, 1, 3), dtype=np.uint8)

def crop_eye(frame, eye_region):
    # Ensure eye_region is a valid numpy array of shape (N, 2) and dtype int32
    if eye_region is None or len(eye_region) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    eye_region = np.array(eye_region, dtype=np.int32)
    if eye_region.ndim != 2 or eye_region.shape[0] < 3 or eye_region.shape[1] != 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if 3 < radius < 80:
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

def estimate_gaze_direction(eye_img):
    if eye_img.size == 0:
        return "center"
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pupil = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(pupil)
        h, w = eye_img.shape[:2]
        x_norm = x / w
        y_norm = y / h
        # Widen all regions for robustness
        if x_norm < 0.35:
            return "left"
        elif x_norm > 0.65:
            return "right"
        elif y_norm < 0.45:
            return "up"
        elif y_norm > 0.65:
            return "down"
        else:
            return "center"
    return "center"

def multi_stage_calibration(cap, detector, predictor):
    directions = [
        ("straight", "Look straight ahead"),
        ("left",     "Turn your gaze to the left"),
        ("right",    "Turn your gaze to the right"),
        ("up",       "Look up"),
        ("down",     "Look down")
    ]
    FRAMES_PER_DIR = 15
    MAX_DURATION   = 10  # seconds max per direction
    TOLERANCE      = 0.25  # allow ±25% around each region

    calibration = {}
    cv2.namedWindow("Calibration")
    speak("Starting quick calibration. Follow the prompts.")

    for dir_key, prompt in directions:
        speak_async(prompt)
        start_t = time.time()
        accepted = 0
        sum_Lr = sum_Rr = sum_Lx = sum_Ly = sum_Rx = sum_Ry = 0.0

        while accepted < FRAMES_PER_DIR and (time.time() - start_t) < MAX_DURATION:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # draw instruction and progress
            cv2.putText(frame, f"{prompt} ({accepted}/{FRAMES_PER_DIR})",
                        (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) == 27:
                break

            if not faces:
                continue

            lm = predictor(gray, faces[0])
            # extract & crop eyes
            le_img = crop_eye(frame, [(lm.part(p).x, lm.part(p).y) for p in range(36,42)])
            re_img = crop_eye(frame, [(lm.part(p).x, lm.part(p).y) for p in range(42,48)])
            Lr = detect_pupil_size(le_img)
            Rr = detect_pupil_size(re_img)
            lx, ly = get_pupil_norm(le_img)
            rx, ry = get_pupil_norm(re_img)

            # normalize gaze
            def in_region(val, target):
                return abs(val - target) <= TOLERANCE

            gaze = estimate_gaze_direction(le_img), estimate_gaze_direction(re_img)
            ok_gaze = False
            if dir_key == "straight":
                ok_gaze = ("center" in gaze)
            elif dir_key in gaze:
                ok_gaze = True
            # allow slight center for up/down
            if dir_key in ("up","down") and "center" in gaze:
                ok_gaze = True

            # accept if gaze roughly correct and pupil sizes plausible
            if ok_gaze and 1 < Lr < 50 and 1 < Rr < 50:
                sum_Lr += Lr; sum_Rr += Rr
                sum_Lx += lx; sum_Ly += ly
                sum_Rx += rx; sum_Ry += ry
                accepted += 1

        # average values (or zeros if none)
        if accepted > 0:
            calibration[dir_key] = {
                "left":   sum_Lr/accepted,
                "right":  sum_Rr/accepted,
                "left_x": sum_Lx/accepted,
                "left_y": sum_Ly/accepted,
                "right_x":sum_Rx/accepted,
                "right_y":sum_Ry/accepted
            }
        else:
            # fallback to center values
            calibration[dir_key] = calibration.get("straight", {
                "left":0,"right":0,"left_x":0.5,"left_y":0.5,"right_x":0.5,"right_y":0.5
            })

    cv2.destroyWindow("Calibration")
    speak("Calibration complete.")
    save_calibration(calibration)
    return calibration

#def show_instruction(text):
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
   
    plt.savefig(graph_file)
    plt.close()

def show_session_summary(log_file, graph_file):
    import pandas as pd
    df = pd.read_csv(log_file)
    total_events = df["event"].value_counts().to_dict()
    avg_left = df["left_pupil"].mean()
    avg_right = df["right_pupil"].mean()

    root = tk.Tk()
    root.title("Session Summary")

    summary = f"Session Summary\n\n"
    for event, count in total_events.items():
        summary += f"{event}: {count}\n"
    summary += f"\nAverage Left Pupil: {avg_left:.2f}\nAverage Right Pupil: {avg_right:.2f}"

    tk.Label(root, text=summary, font=("Arial", 12)).pack(pady=10)

    # Show the graph image
    img = Image.open(graph_file)
    img = img.resize((600, 200))
    photo = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=photo)
    panel.image = photo
    panel.pack(pady=10)

    tk.Button(root, text="Close", command=root.destroy).pack(pady=10)
    root.mainloop()

def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    length = 100
    color = (0, 255, 255)
    thickness = 8
    if direction == "straight":
        cv2.arrowedLine(frame, (center[0], center[1]+length//2), (center[0], center[1]-length//2), color, thickness, tipLength=0.5)
    elif direction == "left":
        cv2.arrowedLine(frame, (center[0]+length//2, center[1]), (center[0]-length//2, center[1]), color, thickness, tipLength=0.5)
    elif direction == "right":
        cv2.arrowedLine(frame, (center[0]-length//2, center[1]), (center[0]+length//2, center[1]), color, thickness, tipLength=0.5)
    elif direction == "up":
        cv2.arrowedLine(frame, (center[0], center[1]+length//2), (center[0], center[1]-length//2), color, thickness, tipLength=0.5)
    elif direction == "down":
        cv2.arrowedLine(frame, (center[0], center[1]-length//2), (center[0], center[1]+length//2), color, thickness, tipLength=0.5)
    # No return needed, modifies frame in-place

def draw_target_overlay(eye_img, pupil_center=None):
    h, w = eye_img.shape[:2]
    center = (w // 2, h // 2) if pupil_center is None else (int(pupil_center[0]), int(pupil_center[1]))
    color = (255, 255, 255)
    thickness = 2

    # Draw crosshairs
    cv2.line(eye_img, (center[0], 0), (center[0], h), color, thickness)
    cv2.line(eye_img, (0, center[1]), (w, center[1]), color, thickness)

    # Draw concentric circles
    for r in range(30, min(w, h)//2, 30):
        cv2.circle(eye_img, center, r, color, 1)

    # Draw a small central box
    box_size = 20
    cv2.rectangle(eye_img, (center[0] - box_size//2, center[1] - box_size//2),
                  (center[0] + box_size//2, center[1] + box_size//2), color, 1)

    # Draw a small central cross
    cv2.drawMarker(eye_img, center, color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    return eye_img

def draw_dynamic_pupil_overlay(eye_img, pupil_radius, max_radius=None):
    h, w = eye_img.shape[:2]
    center = (w // 2, h // 2)
    color = (0, 255, 0)
    base_color = (255, 255, 255)
    thickness = 2

    # Draw static concentric circles (for reference)
    if max_radius is None:
        max_radius = min(w, h) // 2 - 10
    step = max_radius // 4
    for r in range(step, max_radius + 1, step):
        cv2.circle(eye_img, center, r, base_color, 1)

    # Draw the current pupil size as a colored circle
    if pupil_radius > 0:
        cv2.circle(eye_img, center, int(pupil_radius), color, thickness)

    # Draw crosshairs
    cv2.line(eye_img, (center[0], 0), (center[0], h), base_color, 1)
    cv2.line(eye_img, (0, center[1]), (w, center[1]), base_color, 1)

    return eye_img

def render_live_graph(times, left_pupils, right_pupils):
    # create the figure
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.plot(times, left_pupils, label="Left")
    ax.plot(times, right_pupils, label="Right")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Size")
    ax.set_title("Live Pupil Size")
    fig.tight_layout()

    # draw on Agg canvas
    canvas = FigureCanvas(fig)
    canvas.draw()

    # grab the raw ARGB buffer, convert to (H, W, 4)
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    buf = buf.reshape((height, width, 4))

    # drop the alpha channel: take R,G,B channels only
    img = buf[:, :, 1:4]

    plt.close(fig)
    return img


def get_pupil_norm(eye_img):
    if eye_img.size == 0:
        return 0.5, 0.5
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pupil = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(pupil)
        h, w = eye_img.shape[:2]
        return x / w, y / h
    return 0.5, 0.5


# --- MAIN APP ---
cap = cv2.VideoCapture(0)

# Calibration: load or run
calibration_data = load_calibration()
if not calibration_data:
    # Calibration phase
    # calibration_frames = 50
    # baseline = []
    # cv2.namedWindow("Calibration")
    # for i in range(calibration_frames):
    #     ret, frame = cap.read()
    #     if not ret or frame is None or frame.size == 0:
    #         continue  # Skip this iteration if frame is invalid
    #
    #     frame = cv2.flip(frame, 1)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = detector(gray)
    #     if not faces:
    #         cv2.putText(frame, "Align your face...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #         cv2.imshow("Calibration", frame)
    #         cv2.waitKey(1)
    #         continue
    #     face = faces[0]
    #     landmarks = predictor(gray, face)
    #
    #     left_eye_img = get_eye_region(frame, [36,37,38,39,40,41], landmarks)
    #     right_eye_img = get_eye_region(frame, [42,43,44,45,46,47], landmarks)
    #
    #     left_radius = detect_pupil_size(left_eye_img)
    #     right_radius = detect_pupil_size(right_eye_img)
    #     if left_radius > 0 and right_radius > 0:
    #         baseline.append((left_radius + right_radius) / 2)
    #
    #     cv2.putText(frame, f"Calibration: Look straight ahead ({i+1}/{calibration_frames})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    #     cv2.imshow("Calibration", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     time.sleep(0.05)
    # cv2.destroyWindow("Calibration")
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
times, left_pupils, right_pupils = [], [], []
calibration_mode = False
button_state = {"pause": False, "stop": False}

paused = False

def mouse_callback(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        if 1050 <= x <= 1150 and 20 <= y <= 70:
            paused = not paused
            if paused:
                speak("Session paused. Press Play to resume.")
            else:
                speak("Session resumed.")
        elif 1170 <= x <= 1270 and 20 <= y <= 70:
            speak("Session stopped.")
            cv2.destroyAllWindows()
            os._exit(0)

cv2.namedWindow("Dashboard")
cv2.setMouseCallback("Dashboard", mouse_callback)

SEIZURE_FRAMES_THRESHOLD = 30  # e.g., 1 second if running at 30 FPS
abnormal_counter = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - start_time  # <-- Add this line
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # --- Always define these, even if no face is found ---
        left_eye_zoom = np.zeros((100, 200, 3), dtype=np.uint8)
        right_eye_zoom = np.zeros((100, 200, 3), dtype=np.uint8)

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
                left_eye_points_coords = [(landmarks.part(p).x, landmarks.part(p).y) for p in left_eye_points]
                right_eye_points_coords = [(landmarks.part(p).x, landmarks.part(p).y) for p in right_eye_points]
                left_eye_region = np.array(left_eye_points_coords, dtype=np.int32)
                right_eye_region = np.array(right_eye_points_coords, dtype=np.int32)
                left_eye = crop_eye(frame, left_eye_region)
                right_eye = crop_eye(frame, right_eye_region)

                if left_eye is not None and left_eye.size > 0:
                    left_eye_zoom = cv2.resize(left_eye, (200, 100))
                else:
                    left_eye_zoom = np.zeros((100, 200, 3), dtype=np.uint8)

                right_eye = cv2.resize(right_eye, (200, 100))
                left_pupil = detect_pupil_size(left_eye)
                right_pupil = detect_pupil_size(right_eye)

                x_norm_left, y_norm_left = get_pupil_norm(left_eye)
                x_norm_right, y_norm_right = get_pupil_norm(right_eye)
                print(f"y_norm_left={y_norm_left:.2f}, y_norm_right={y_norm_right:.2f}")

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
                    abnormal_counter += 1
                    if abnormal_counter >= SEIZURE_FRAMES_THRESHOLD:
                        # Only now trigger a seizure event
                        cv2.rectangle(frame, (80, 420), (600, 470), (0, 0, 255), -1)
                        cv2.putText(frame, "⚠️ Possible Seizure Detected", (100, 455),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                        if not warning_played:
                            speak("Possible seizure detected. Please seek help.")
                            warning_played = True
                        event = "SEIZURE"
                else:
                    abnormal_counter = 0
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

        # --- Live graph rendering ---
        times.append(elapsed)
        left_pupils.append(left_pupil)
        right_pupils.append(right_pupil)
        graph_view = render_live_graph(times, left_pupils, right_pupils)
        graph_view_resized = cv2.resize(graph_view, (320, 240))
        
        


        # --- Dashboard view ---
        face_view = frame
        face_view_resized = cv2.resize(face_view, (320, 240))
        left_eye_view = left_eye_zoom
        left_eye_resized = cv2.resize(left_eye_view, (320, 120))
        right_eye_view = right_eye_zoom
        right_eye_resized = cv2.resize(right_eye_view, (320, 120))
        graph_view_resized = cv2.resize(graph_view, (320, 240))

        top_row = np.hstack((face_view_resized, graph_view_resized))      # shape: (240, 640, 3)
        bottom_row = np.hstack((left_eye_resized, right_eye_resized))     # shape: (120, 640, 3)
        dashboard = np.vstack((top_row, bottom_row))                      # shape: (360, 640, 3)

        dashboard_large = cv2.resize(dashboard, (1280, 720))  # or (1920, 1080) for full HD
        cv2.imshow("Dashboard", dashboard_large)
        cv2.moveWindow("Dashboard", 50, 50)  # Optional: move to near top-left

        # Add this before showing the dashboard
        instructions = ""
        if not paused:
            if calibration_mode:
                instructions = "Calibration: Follow the arrow. Press P=Pause, S=Stop, M=Mute"
            else:
                instructions = "Session: Look straight. P=Pause, S=Stop, M=Mute"
        cv2.putText(dashboard_large, instructions, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw Pause/Play and Stop buttons
        cv2.rectangle(dashboard_large, (1050, 20), (1150, 70), (40, 40, 40), -1)  # Pause/Play
        cv2.rectangle(dashboard_large, (1170, 20), (1270, 70), (40, 40, 40), -1)  # Stop

        if paused:
            cv2.putText(dashboard_large, "Play", (1070, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            cv2.putText(dashboard_large, "Pause", (1060, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(dashboard_large, "Stop", (1190, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == ord('p'):  # P key
        paused = not paused
        if paused:
            speak("Session paused. Press P to resume.")
        else:
            speak("Session resumed.")
    elif key == ord('c'):  # C key
        event = "SESSION_CANCELLED"
        cv2.rectangle(frame, (60, 420), (700, 470), (0, 0, 0), -1)
        cv2.putText(frame, "Session Cancelled", (80, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.imshow("Dashboard", frame)
        log_file.close()
        break
    elif key == ord('s'):  # S key (stop)
        cv2.rectangle(frame, (60, 420), (700, 470), (0, 0, 0), -1)
        cv2.putText(frame, "Session Stopped", (80, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.imshow("Dashboard", frame)
        log_file.close()
        time.sleep(2)
        # Ask for doctor mode
        doctor_mode = False
        while True:
            speak("Session stopped. Do you want to enter doctor mode?")
            response = input("Enter doctor mode? (y/n): ")

            # Normalize response
            response = response.strip().lower()

            if response == "y":
                doctor_mode = True
                break
            elif response == "n":
                doctor_mode = False
                break
            else:
                speak("Invalid response. Please answer with 'yes' or 'no'.")

        if doctor_mode:
            # Doctor mode: show detailed graph and stats
            cv2.destroyWindow("Dashboard")
            df = pd.read_csv(LOG_FILE)
            plt.figure(figsize=(10, 5))
            plt.plot(df["timestamp"], df["left_pupil"], label="Left Pupil")
            plt.plot(df["timestamp"], df["right_pupil"], label="Right Pupil")
            plt.xlabel("Time")
            plt.ylabel("Pupil Size")
            plt.title("Pupil Size Over Time (Doctor Mode)")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        break
    elif key == ord('m'):  # M key (mute/unmute)
        voice_enabled = not voice_enabled
        if voice_enabled:
            speak("Voice guidance enabled.")
        else:
            speak("Voice guidance disabled.")

cap.release()
cv2.destroyAllWindows()

# Finalize the log file
log_file.close()

# Generate graph and show session summary
plot_and_save_graph(LOG_FILE, GRAPH_FILE)
show_session_summary(LOG_FILE, GRAPH_FILE)
