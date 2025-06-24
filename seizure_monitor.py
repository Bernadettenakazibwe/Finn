import cv2
import dlib
import numpy as np
import pyttsx3
import time
import csv
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
CALIB_SAMPLES       = 30      # frames for calibration
SMOOTH_WINDOW       = 5       # frames for moving average
DILATION_FACTOR     = 1.4     # 40% dilation above baseline
ASYM_FACTOR         = 1.2     # 20% asymmetry
DEBOUNCE_FRAMES     = 5       # consecutive frames needed
LOG_CSV_PATH        = "session_log.csv"

# === Text-to-Speech Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Pupil Detection with Adaptive Preprocessing ===
def detect_pupil_size(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)                        # normalize brightness
    gray = cv2.GaussianBlur(gray, (7, 7), 0)             # reduce noise
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    # remove small blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, None
    pupil = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(pupil)
    if radius < 3 or radius > max(eye_img.shape)/2:
        return 0, None
    return radius, pupil

# === Crop Eye by Landmark Region ===
def crop_eye(frame, region):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(region)
    return eye[y:y+h, x:x+w], (x, y)

# === Simple Straight-Ahead Calibration ===
def simple_calibration(cap, detector, predictor):
    speak("Calibration: please look straight at the camera.")
    sum_L = sum_R = 0
    count = 0

    while count < CALIB_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            lm = predictor(gray, faces[0])
            # eye landmarks
            left_region  = np.array([(lm.part(i).x, lm.part(i).y) for i in range(36, 42)])
            right_region = np.array([(lm.part(i).x, lm.part(i).y) for i in range(42, 48)])
            le, _ = crop_eye(frame, left_region)
            re, _ = crop_eye(frame, right_region)

            Lr, _ = detect_pupil_size(le)
            Rr, _ = detect_pupil_size(re)
            if 1 < Lr < 80 and 1 < Rr < 80:
                sum_L += Lr
                sum_R += Rr
                count += 1

        cv2.putText(frame,
            f"Calibrating: {count}/{CALIB_SAMPLES}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyWindow("Calibration")
    baseline_L = sum_L / max(count,1)
    baseline_R = sum_R / max(count,1)
    speak("Calibration complete.")
    return baseline_L, baseline_R

# === Main Monitoring Loop ===
def main():
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)

    # 1) Calibrate
    baseline_L, baseline_R = simple_calibration(cap, detector, predictor)

    # 2) Prepare smoothing & logging
    left_hist  = deque(maxlen=SMOOTH_WINDOW)
    right_hist = deque(maxlen=SMOOTH_WINDOW)
    anomaly_count = 0

    # Open CSV log
    logf = open(LOG_CSV_PATH, "w", newline="")
    logger = csv.writer(logf)
    logger.writerow(["timestamp","L_raw","R_raw","L_norm","R_norm","L_smooth","R_smooth","anomaly"])

    speak("Starting monitoring. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Initialize variables to safe defaults
        L_raw = R_raw = 0
        L_norm = R_norm = L_smooth = R_smooth = 0
        is_anomaly = False
        pupil_contour = None
        offset = (0,0)

        if faces:
            lm = predictor(gray, faces[0])
            left_region  = np.array([(lm.part(i).x, lm.part(i).y) for i in range(36,42)])
            right_region = np.array([(lm.part(i).x, lm.part(i).y) for i in range(42,48)])
            le, offL = crop_eye(frame, left_region)
            re, offR = crop_eye(frame, right_region)

            L_raw, cntL = detect_pupil_size(le)
            R_raw, cntR = detect_pupil_size(re)

            # normalize
            L_norm = L_raw / (baseline_L + 1e-6)
            R_norm = R_raw / (baseline_R + 1e-6)

            # smooth
            left_hist.append(L_norm)
            right_hist.append(R_norm)
            L_smooth = sum(left_hist)/len(left_hist)
            R_smooth = sum(right_hist)/len(right_hist)

            # anomaly logic
            over_dilate = (L_smooth > DILATION_FACTOR) or (R_smooth > DILATION_FACTOR)
            asymmetry   = abs(L_smooth - R_smooth) > ASYM_FACTOR
            is_anomaly  = over_dilate or asymmetry

            anomaly_count = anomaly_count+1 if is_anomaly else 0
            if anomaly_count >= DEBOUNCE_FRAMES:
                cv2.putText(frame,
                    "⚠️ Possible Seizure Detected",
                    (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            # overlay pupil contours for debug
            if cntL is not None:
                pupil_contour = cntL + offL
                cv2.drawContours(frame, [pupil_contour], -1, (0,255,0), 1)
            if cntR is not None:
                pupil_contour = cntR + offR
                cv2.drawContours(frame, [pupil_contour], -1, (0,255,0), 1)

            # show zoomed eyes
            zoomL = cv2.resize(le, (200,100))
            zoomR = cv2.resize(re, (200,100))
            cv2.putText(zoomL,  f"{L_smooth:.2f}×", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.putText(zoomR,  f"{R_smooth:.2f}×", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.imshow("Zoomed Eyes", np.hstack((zoomL, zoomR)))

        # log data
        logger.writerow([
            time.time(), L_raw, R_raw,
            f"{L_norm:.2f}", f"{R_norm:.2f}",
            f"{L_smooth:.2f}", f"{R_smooth:.2f}",
            is_anomaly
        ])

        cv2.imshow("Seizure Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    logf.close()
    speak("Session ended. Goodbye.")
    
        



    # 1) Load and clean data
    df = pd.read_csv('session_log.csv')
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]
    df = df[~((df['L_smooth']==0) & (df['R_smooth']==0))]

    # 2) Identify anomaly spans
    anomaly = df['anomaly'].astype(bool)
    # We’ll create spans of contiguous True segments
    spans = []
    in_span = False
    for t, flag in zip(df['time_s'], anomaly):
        if flag and not in_span:
            start = t; in_span = True
        elif not flag and in_span:
            spans.append((start, t)); in_span = False
    if in_span:  # close final span
        spans.append((start, df['time_s'].iloc[-1]))

    # 3) Plot smoothed pupil sizes
    plt.figure(figsize=(10,4))
    plt.plot(df['time_s'], df['L_smooth'], label='Left', alpha=0.8)
    plt.plot(df['time_s'], df['R_smooth'], label='Right', alpha=0.8)

    # 4) Shade anomaly spans
    for (t0, t1) in spans:
        plt.axvspan(t0, t1, color='red', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Smoothed Pupil Size (× baseline)')
    plt.title('Pupil Size Over Time with Seizure Alerts Shaded')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.axhline(DILATION_FACTOR, color='red', linestyle=':')

    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
