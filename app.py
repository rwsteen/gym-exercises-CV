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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("stgcn_model.pth", map_location=device))
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
        cap = cv2.VideoCapture(1)
    else:
        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")

    frame_count = 0

    # Process video frames and extract pose landmarks with MediaPipe
    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        T = 16 # Number of frames to process for each sample (for model input)
        V = 13 # Number of joints (Penn Action dataset)
        C = 2 # only x and y coordinates as channels

        joint_buffer = [] # buffer to hold joint data for T frames
        pred_action = "N/A"
        pred_count = 0
        prev_phase = "N/A"
        frame_skip = 4 # skip frames to reduce computation time
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 240))
            results = pose.process(image)

            if results.pose_landmarks:
                joints = []
                row = [frame_count]

                for lm in results.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                
                joints = extract_penn_joints(results, V, C) # (13, 2)

                # Add joints to buffer
                joints = np.array(joints) # (V, 2)
                joint_buffer.append(joints)

                if len(joint_buffer) > T:
                    joint_buffer.pop(0) # keep only the last T frames

                if len(joint_buffer) == T:
                    x = torch.from_numpy(np.array(joint_buffer).transpose(2,0,1)[None,...,None]).float().to(device) # (1, C, T, V, 1)
                    with torch.no_grad():
                        action, phase = model(x)

                    pred_action = exercise_labels[torch.argmax(action).item()]
                    
                    # get current phase from model
                    phase_np = phase.squeeze().cpu().numpy()  # (T,)
                    phase = phase_np[-1]
                    print("phase", phase)

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

                # Draw black rectangles behind the text
                cv2.rectangle(frame, (5, 5), (250, 50), (0,0,0), -1)   # rectangle for exercise
                cv2.rectangle(frame, (5, 55), (150, 100), (0,0,0), -1) # rectangle for count

                # Overlay text on top
                cv2.putText(frame, f"Exercise: {pred_action}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Count: {pred_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            frame_count += 1

            frame_placeholder.image(frame, channels="BGR")

    cap.release()