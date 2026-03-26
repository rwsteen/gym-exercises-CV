# run app with "streamlit run app.py"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import csv
from model.model import STGCN
from form import form_analysis

# load model
NUM_CLASSES = 2
NUM_JOINTS = 13
model = STGCN(num_class=NUM_CLASSES, num_point=NUM_JOINTS, in_channels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Penn Action dataset labels
exercise_labels = [
    "squat",
    "pushup",
]

# MediaPipe pose landmarks to Penn Action joints mapping
PENN_JOINTS = [
    0,   # head 0
    11,  # left_shoulder 1
    12,  # right_shoulder 2
    13,  # left_elbow 3
    14,  # right_elbow 4
    15,  # left_wrist 5
    16,  # right_wrist 6
    23,  # left_hip 7
    24,  # right_hip 8
    25,  # left_knee 9
    26,  # right_knee 10
    27,  # left_ankle 11
    28   # right_ankle 12
]

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_penn_joints(results, prev_joints=None, V=13, C=3):
    joints = np.zeros((V, C))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        for i, mp_idx in enumerate(PENN_JOINTS):
            lm = landmarks[mp_idx]

            if lm.visibility < 0.5:  # threshold
                if prev_joints is not None:
                    joints[i] = prev_joints[i]  # fallback to previous frame
                else:
                    joints[i] = [0, 0, 0]
            else:
                joints[i] = [lm.x, lm.y, lm.visibility]
    
    # normalize root center
    left_hip = joints[7]
    right_hip = joints[8]
    hip_midpoint = (left_hip[:2] + right_hip[:2]) / 2.0
    joints[:, :2] -= hip_midpoint

    # scale normalize
    max_val = np.max(np.abs(joints[:, :2]))
    if max_val > 1e-6:
        joints[:, :2] /= max_val


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
        C = 3 # only x and y coordinates as channels

        joint_buffer = [] # buffer to hold joint data for T frames
        pred_action = "N/A"
        pred_count = 0
        pred_shallow_rep = 0
        deepnes = 0
        prev_state = "up" # for counting reps (assumes starting in up position)
        frame_skip = 2 # skip frames to reduce computation time
        back_hollow = 0
        rounded_back = 0
        perfect_form = 0
        forward_bend2 = 0
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
                
                joints = extract_penn_joints(results) # (13, 3)
                
                landmarks = results.pose_landmarks.landmark

                left_vis_pu = (
                    landmarks[11].visibility +  # left shoulder
                    landmarks[13].visibility    # left elbow
                )
                
                right_vis_pu = (
                    landmarks[12].visibility +  # right shoulder
                    landmarks[14].visibility    # right elbow
                )
                
                left_vis_s = (
                    landmarks[23].visibility +  # left hip
                    landmarks[25].visibility    # left knee
                )
                
                right_vis_s = (
                    landmarks[24].visibility +  # right hip
                    landmarks[26].visibility    # right knee
                )

                # Add joints to buffer
                joints = np.array(joints) # (V, 3)
                joint_buffer.append(joints)

                if len(joint_buffer) > T:
                    joint_buffer.pop(0) # keep only the last T frames

                if len(joint_buffer) == T:
                    x = torch.from_numpy(np.array(joint_buffer).transpose(2,0,1)[None,...,None]).float().to(device) # (1, C, T, V, 1)
                    with torch.no_grad():
                        action = model(x)

                    pred_action = exercise_labels[torch.argmax(action).item()]
                    
                    if pred_action == "squat":
                        # determine which side is more visible
                        if left_vis_s > right_vis_s:
                            visible_side = "left"
                        else:
                            visible_side = "right"
                        
                        deep_enough, forward_bend = form_analysis(pred_action, prev_state, x, visible_side)
                        deepnes += deep_enough
                        if forward_bend == 0:
                            perfect_form += 1
                        else:
                            forward_bend2 += 1
                        
                        left_knee_angle = calculate_angle(joints[7,:2], joints[9,:2], joints[11,:2])
                        right_knee_angle = calculate_angle(joints[8,:2], joints[10,:2], joints[12,:2])
                        knee_angle = (left_knee_angle + right_knee_angle) / 2.0
                        print(f"Knee angle: {knee_angle:.2f}")

                        if knee_angle < 70 and prev_state == "up":
                            prev_state = "down"
                        elif knee_angle > 160 and prev_state == "down":
                            pred_count += 1
                            prev_state = "up"
                            if deepnes == 0:
                                pred_shallow_rep += 1
                            deepnes = 0

                    elif pred_action == "pushup":
                        # determine which side is more visible
                        if left_vis_pu > right_vis_pu:
                            visible_side = "left"
                        else:
                            visible_side = "right"
                            
                        deep_enough, back_form = form_analysis(pred_action, prev_state, x, visible_side)
                        deepnes += deep_enough
                        if back_form > 0:
                            rounded_back += 1
                        elif back_form < 0:
                            back_hollow += 1
                        else:
                            perfect_form += 1
                        
                        left_elbow_angle = calculate_angle(joints[1,:2], joints[3,:2], joints[5,:2])
                        right_elbow_angle = calculate_angle(joints[2,:2], joints[4,:2], joints[6,:2])
                        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
                        print(f"Elbow angle: {elbow_angle:.2f}")

                        if elbow_angle < 90 and prev_state == "up":
                            prev_state = "down"
                        elif elbow_angle > 120 and prev_state == "down":
                            pred_count += 1
                            prev_state = "up"
                            if deepnes == 0:
                                pred_shallow_rep += 1
                            deepnes = 0
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Draw black rectangles behind the text
                cv2.rectangle(frame, (5, 5), (300, 50), (0,0,0), -1)   # rectangle for exercise
                cv2.rectangle(frame, (5, 55), (200, 100), (0,0,0), -1) # rectangle for count

                # Overlay text on top
                cv2.putText(frame, f"Exercise: {pred_action}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Count: {pred_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Shallow: {pred_shallow_rep}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if pred_action == "pushup":
                    cv2.putText(frame, f"Hollow back: {back_hollow}, Rounded back: {rounded_back}, Good form: {perfect_form}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if pred_action == "squat":
                    cv2.putText(frame, f"Forward bend: {forward_bend2}, Good form: {perfect_form}", (10, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


            frame_count += 1

            frame_placeholder.image(frame, channels="BGR")

    cap.release()