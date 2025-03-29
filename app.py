import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained CNN model
model = load_model("sign_language_model.h5")

# Define class names (digits 0-9 + lowercase letters a-z)
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Encode class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize OpenCV webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe expects RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around hand
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Expand bounding box slightly
            x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)
            x_max, y_max = min(frame.shape[1], x_max + 20), min(frame.shape[0], y_max + 20)

            # Extract the ROI (hand region)
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.shape[0] > 0 and roi.shape[1] > 0:  # Ensure the ROI is valid
                roi = cv2.resize(roi, (64, 64))
                roi = roi / 255.0  # Normalize pixel values
                roi = np.expand_dims(roi, axis=0)  # Add batch dimension

                # Make prediction
                preds = model.predict(roi)
                pred_class = np.argmax(preds)
                pred_label = label_encoder.inverse_transform([pred_class])[0]

                # Display prediction
                cv2.putText(frame, f"Prediction: {pred_label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
