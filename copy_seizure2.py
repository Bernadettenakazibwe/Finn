import cv2
import dlib
import numpy as np

# Load face detector and facial landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to get the eye region
def get_eye_region(landmarks, eye_points):
    return np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points], np.int32)

# Crop the eye and return the region
def crop_eye(frame, eye_region):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(eye_region)
    cropped = eye[y:y+h, x:x+w]
    return cropped


# Estimate pupil size by detecting dark area
def detect_pupil_size(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pupil = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(pupil)
        return radius
    return 0

# Webcam
cap = cv2.VideoCapture(0)

prev_left, prev_right = 0, 0
threshold_change = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]

        left_eye_region = get_eye_region(landmarks, left_eye_points)
        right_eye_region = get_eye_region(landmarks, right_eye_points)

        left_eye = crop_eye(frame, left_eye_region)
        right_eye = crop_eye(frame, right_eye_region)

        # Resize for zoom effect
        left_eye_zoom = cv2.resize(left_eye, (200, 100))
        right_eye_zoom = cv2.resize(right_eye, (200, 100))

        # Pupil sizes
        left_pupil = detect_pupil_size(left_eye)
        right_pupil = detect_pupil_size(right_eye)

        sudden_change_left = abs(left_pupil - prev_left) > threshold_change
        sudden_change_right = abs(right_pupil - prev_right) > threshold_change
        asymmetry = abs(left_pupil - right_pupil) > 6

        # Display warning if condition is met
        if sudden_change_left or sudden_change_right or asymmetry:
            cv2.putText(frame, "⚠️ Possible Seizure Detected", (100, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        prev_left = left_pupil
        prev_right = right_pupil

        # Show zoomed eyes and pupil sizes
        eye_view = np.hstack((left_eye_zoom, right_eye_zoom))
        cv2.putText(eye_view, f"Left: {left_pupil:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(eye_view, f"Right: {right_pupil:.2f}", (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Zoomed Eyes View", eye_view)

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
