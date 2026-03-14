# run app with "streamlit run app.py"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import csv
from model.model import STGCN

# load model
NUM_CLASSES = 2
NUM_JOINTS = 13
model = STGCN(num_class=NUM_CLASSES, num_point=NUM_JOINTS)
model.load_state_dict(torch.load("stgcn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Penn Action dataset labels
exercise_labels = [
    "squat",
    "pushup"
]

# MediaPipe pose landmarks to Penn Action joints mapping
PENN_JOINTS = [
    0,   # head
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28   # right_ankle
]

def buffer_to_tensor(buffer):
    data = np.array(buffer) # (T, V, C)
    data = data.transpose(2, 0, 1) # (C, T, V)
    data = np.expand_dims(data, axis=0) # (1, C, T, V)
    data = np.expand_dims(data, axis=-1) # (1, C, T, V, 1)

    return torch.tensor(data, dtype=torch.float32)

def extract_penn_joints(results, V=13, C=2):
    joints = np.zeros((V, C)) # (V, C)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        for i, mp_idx in enumerate(PENN_JOINTS):
            joints[i] = [landmarks[mp_idx].x, landmarks[mp_idx].y]
    
    return joints
    

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("Exercise Detection App")

video_source = st.selectbox(
    "Select Input Source",
    ("Webcam", "Video File")
)

video_file = None

if video_source == "Video File":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

start = st.button("Start")

frame_placeholder = st.empty()

if start:

    # Set up video capture based on the selected source
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")

    # Set up CSV file for writing landmarks (joints)
    file = open("landmarks.csv", "w", newline="")
    writer = csv.writer(file)

    # Write header to CSV
    header = ["frame"]
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
    writer.writerow(header)

    frame_count = 0

    # Process video frames and extract pose landmarks with MediaPipe
    with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        T = 64 # Number of frames to process for each sample (for model input)
        V = 13 # Number of joints (Penn Action dataset)
        C = 2 # only x and y coordinates as channels

        joint_buffer = [] # buffer to hold joint data for T frames
        pred_action = "N/A"
        pred_count = 0
        prev_phase = "N/A"
        T_smooth = 5
        phase_buffer = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                joints = []
                row = [frame_count]

                for lm in results.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                
                joints = extract_penn_joints(results, V, C) # (13, 2)

                # Write landmarks to CSV
                writer.writerow(row)

                # Add joints to buffer
                joints = np.array(joints) # (V, 2)
                joint_buffer.append(joints)

                if len(joint_buffer) > T:
                    joint_buffer.pop(0) # keep only the last T frames

                if len(joint_buffer) == T:
                    x = buffer_to_tensor(joint_buffer) # (1, C, T, V, 1)
                    with torch.no_grad():
                        action, phase = model(x)

                    pred_action = exercise_labels[torch.argmax(action).item()]
                    
                    # get current phase from model (last frame in the window)
                    phase_np = phase.squeeze().cpu().numpy()  # (T,)
                    phase = phase_np[-1]

                    if prev_phase == "N/A":
                        prev_phase = "up" if phase > 0.5 else "down"
                    elif prev_phase == "up" and phase < 0.3:
                        prev_phase = "down"
                    elif prev_phase == "down" and phase > 0.7:
                        pred_count += 1
                        prev_phase = "up"
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(frame, f"Exercise: {pred_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Count: {pred_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            frame_count += 1

            frame_placeholder.image(frame, channels="BGR")

    cap.release()
    file.close()