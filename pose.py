import cv2
import mediapipe as mp
import numpy as np
import mediapipe.python.solutions as solutions
import csv

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_pose = solutions.pose


file = open("landmarks.csv", mode="w", newline="")
writer = csv.writer(file)

# header maken
header = ["frame"]
for i in range(33):
    header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
writer.writerow(header)

frame_count = 0

cap = cv2.VideoCapture("didier.mp4") # mp4 file for video or 0 for webcam
window_name = "MediaPipe Pose"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)
cv2.moveWindow(window_name, 100, 50)
with mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        row = [frame_count]
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
        writer.writerow(row)

    frame_count += 1

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow(window_name, cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
file.close()
