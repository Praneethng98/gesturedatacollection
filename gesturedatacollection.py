import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import pickle


# ------------------------------------
# Gesture Mappings (Static & Dynamic)
# ------------------------------------
gesture_mapping_static = {
    ord('1'): "Compress",
    ord('2'): "Medical Thread on a Spool",
    ord('3'): "Medical Thread Loose",
    ord('4'): "Plier Backhaus",
    ord('5'): "Hemostatic Forceps",
    ord('6'): "Kelly Hemostatic Forceps",
}

gesture_mapping_dynamic = {
    ord('7'): "Farabeuf Retractor",
    ord('8'): "Bistouri",
    ord('9'): "Needle Holder",
    ord('a'): "Valve Doyen",
    ord('b'): "Allis Clamp",
    ord('c'): "Anatomical Tweezers",
    ord('d'): "Rats Tooth Forceps",
    ord('e'): "Scissors"
}

# Data holders
static_data = []
static_labels = []
dynamic_data = []
dynamic_labels = []

# Counters
static_counts = defaultdict(int)
dynamic_counts = defaultdict(int)

# Recording settings
sequence_length = 25
recording_dynamic = False
current_sequence = []
current_dynamic_label = ""
capture_mode = "static"  # default mode

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)
print(cap.isOpened())
gesture_display = ""

print("Press TAB to toggle between static and dynamic modes.")
print("Press keys 1-6 (static) or 7-9, a-e (dynamic) to capture gestures.")
print("Press 'q' to quit and save data.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        current_landmarks = []
        for lm in hand_landmarks.landmark:
            current_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show capture mode
    cv2.putText(frame, f"Mode: {capture_mode.upper()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show current gesture being captured
    if gesture_display:
        cv2.putText(frame, gesture_display, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display per-gesture counts
    y_offset = 130
    cv2.putText(frame, "Static Gesture Counts:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, (label, count) in enumerate(static_counts.items()):
        cv2.putText(frame, f"{label}: {count}", (10, y_offset + 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    y_offset += 25 + len(static_counts) * 20 + 10
    cv2.putText(frame, "Dynamic Gesture Counts:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, (label, count) in enumerate(dynamic_counts.items()):
        cv2.putText(frame, f"{label}: {count}", (10, y_offset + 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)

    cv2.imshow("Gesture Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('\t'):
        capture_mode = "dynamic" if capture_mode == "static" else "static"
        gesture_display = f"Switched to {capture_mode.upper()} mode"

    elif key in gesture_mapping_static and capture_mode == "static":
        if current_landmarks:
            label = gesture_mapping_static[key]
            static_data.append(current_landmarks)
            static_labels.append(label)
            static_counts[label] += 1
            gesture_display = f"Captured STATIC: {label} ({static_counts[label]})"
            print(gesture_display)
        else:
            gesture_display = "No hand detected for static capture."

    elif key in gesture_mapping_dynamic and capture_mode == "dynamic":
        if not recording_dynamic:
            recording_dynamic = True
            current_sequence = []
            current_dynamic_label = gesture_mapping_dynamic[key]
            gesture_display = f"Started recording DYNAMIC: {current_dynamic_label}"
            print(gesture_display)

    if recording_dynamic:
        if current_landmarks:
            current_sequence.append(current_landmarks)
        if len(current_sequence) >= sequence_length:
            dynamic_data.append(current_sequence.copy())
            dynamic_labels.append(current_dynamic_label)
            dynamic_counts[current_dynamic_label] += 1
            gesture_display = f"Captured DYNAMIC: {current_dynamic_label} ({dynamic_counts[current_dynamic_label]})"
            print(gesture_display)
            recording_dynamic = False

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# -------------------------
# Save Data to Pickle
# -------------------------
data_to_save = {
    'static_data': static_data,
    'static_labels': static_labels,
    'dynamic_data': dynamic_data,
    'dynamic_labels': dynamic_labels
}

with open('gesture_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Data collection complete.")
print(f"Static samples: {len(static_data)} | Dynamic samples: {len(dynamic_data)}")
print("Data saved to: gesture_data.pkl")



