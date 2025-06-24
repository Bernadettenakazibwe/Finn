import cv2
import dlib

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start webcam
cap = cv2.VideoCapture(0)

def get_distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get important points
        mouth_left = landmarks.part(48)
        mouth_right = landmarks.part(54)
        mouth_top = landmarks.part(51)
        mouth_bottom = landmarks.part(57)

        left_eye_top = landmarks.part(37)
        left_eye_bottom = landmarks.part(41)
        right_eye_top = landmarks.part(43)
        right_eye_bottom = landmarks.part(47)

        # Calculate distances
        mouth_width = get_distance(mouth_left, mouth_right)
        mouth_height = get_distance(mouth_top, mouth_bottom)

        left_eye_opening = get_distance(left_eye_top, left_eye_bottom)
        right_eye_opening = get_distance(right_eye_top, right_eye_bottom)

        # Simple thresholds (you might adjust a little based on your webcam)
        mouth_ratio = mouth_height / mouth_width
        eye_opening_avg = (left_eye_opening + right_eye_opening) / 2

        # Emotion rules
        emotion = "Neutral"
        if mouth_ratio > 0.35:
            emotion = "Surprised"
        elif mouth_ratio > 0.25:
            emotion = "Happy"

        # Draw all landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display emotion text
        cv2.putText(frame, f"Emotion: {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Emotion Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
