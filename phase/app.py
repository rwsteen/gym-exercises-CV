# run app with "streamlit run app.py"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import csv
from phase.model import STGCN
from form import form_analysis

# load model
NUM_CLASSES = 2
NUM_JOINTS = 13
model = STGCN(num_class=NUM_CLASSES, num_point=NUM_JOINTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("phase/best_model_phase_final.pth", map_location=device))
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

# Feedback section below the video
feedback_header = st.empty()
feedback_cols_placeholder = st.empty()
detail_placeholder = st.empty()

if start:

    # Set up video capture based on the selected source
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")

    frame_count = 0

    # Process video frames and extract pose landmarks with MediaPipe
    with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        T = 16 # Number of frames to process for each sample (for model input)
        V = 13 # Number of joints (Penn Action dataset)
        C = 2 # only x and y coordinates as channels

        joint_buffer = [] # buffer to hold joint data for T frames
        pred_action = "N/A"
        pred_count = 0
        pred_shallow_rep = 0
        deepness = 0
        prev_phase = "N/A"
        frame_skip = 4 # skip frames to reduce computation time
        back_hollow = 0
        rounded_back = 0
        perfect_form = 0
        forward_bend2 = 0
        feedback = {}
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
                    
                    #'''
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
                    
                    if pred_action == "pushup":
                        # determine which side is more visible
                        if left_vis_pu > right_vis_pu:
                            visible_side = "left"
                        else:
                            visible_side = "right"
                            
                    if pred_action == "squat":
                        # determine which side is more visible
                        if left_vis_s > right_vis_s:
                            visible_side = "left"
                        else:
                            visible_side = "right"
                    
                    if pred_action == "pushup":
                        deep_enough, back_form, feedback = form_analysis(pred_action, prev_phase, x, visible_side)
                        deepness += deep_enough

                        if back_form > 0:
                            rounded_back += 1
                        elif back_form < 0:
                            back_hollow += 1
                        else:
                            perfect_form += 1
                
                    if pred_action == "squat":
                        deep_enough, forward_bend, feedback = form_analysis(pred_action, prev_phase, x, visible_side)
                        deepness += deep_enough

                        if forward_bend == 0:
                            perfect_form += 1
                        else:
                            forward_bend2 += 1
                    #'''

                    if prev_phase == "N/A":
                        prev_phase = "up" if phase > 0.5 else "down"
                    elif prev_phase == "up" and phase < 0.4:
                        prev_phase = "down"
                    elif prev_phase == "down" and phase > 0.6:
                        pred_count += 1
                        prev_phase = "up"
                        if deepness == 0:
                            pred_shallow_rep += 1
                        deepness = 0
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(frame, f"Exercise: {pred_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Count: {pred_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Shallow: {pred_shallow_rep}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if pred_action == "pushup":
                    cv2.putText(frame, f"Hollow back: {back_hollow}, Rounded back: {rounded_back}, Good form: {perfect_form}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if pred_action == "squat":
                    cv2.putText(frame, f"Forward bend: {forward_bend2}, Good form: {perfect_form}", (10, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


            frame_count += 1

            frame_placeholder.image(frame, channels="BGR")

            if feedback:
                feedback_header.markdown(f"### Form Feedback — {pred_action.capitalize()}")
 
                with feedback_cols_placeholder.container():
                    if pred_action == "pushup":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Good Form", perfect_form)
                        with col2:
                            st.metric("Rounded Back", rounded_back)
                        with col3:
                            st.metric("Hollow Back", back_hollow)
 
                        col4, col5, col6 = st.columns(3)
                        with col4:
                            body_angle = feedback.get("body_angle", 0)
                            st.metric("Body Angle", f"{body_angle:.1f}°")
                        with col5:
                            elbow_angle_fb = feedback.get("bottom_elbow_angle")
                            if elbow_angle_fb is not None:
                                st.metric("Bottom Elbow Angle", f"{elbow_angle_fb:.1f}°")
                        with col6:
                            hip_deviation = feedback.get("hip_deviation", 0)
                            st.metric("Hip Deviation", f"{hip_deviation:.4f}")
 
                    elif pred_action == "squat":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Good Form", perfect_form)
                        with col2:
                            st.metric("Forward Lean", forward_bend2)
 
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            lean = feedback.get("torso_lean_angle", 0)
                            st.metric("Torso Lean", f"{lean:.1f}°")
                        with col4:
                            knee_angle_fb = feedback.get("bottom_knee_angle")
                            if knee_angle_fb is not None:
                                st.metric("Bottom Knee Angle", f"{knee_angle_fb:.1f}°")
                        with col5:
                            valgus = feedback.get("knee_valgus", "N/A")
                            valgus_icon = "✅" if valgus == "good" else ("⚠️" if valgus == "mild" else "❌" if valgus == "severe" else "❓")
                            st.markdown(f"{valgus_icon} **Knee valgus:** {valgus}")
    cap.release()