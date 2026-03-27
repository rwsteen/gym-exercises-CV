# Virtual Gym Assistance using Deep Learning

*By: Rogier van Tienen, Rayan Salmi & Ricardo Steen*

*From: Delft University of Technology*

*Date: 23rd of March 2026*

*Group: 1 (The three R's)*

---

### TODO; figures

Figures:

- dataset example frames

## Introduction

In the last years, the popularity of the gym has been increasing. Most of these gyms have employees working in the gym to help people with exercises, but there are also a large number of gyms where no employee is present. This can cause difficulties for people who have just started to work out and have questions about the gym exercises they are performing. To come with a solution, this blog proposes a virtual gym assistant. With new computer vision techniques and pose estimation it becomes easier to track the human movement. The virtual gym assistant uses these new techniques to classify the gym exercise, count the reps, and gives feedback on how the exercise is performed and whether the form is correct. Where in recent papers rule based approaches are used for rep counting, this virtual assistant used training-based rep counting. 
In this blog, we first discuss related work to get an idea of the models that are used now for exercise classification and rep counting. Next, we introduce the dataset used in this study and explain the motivation behind the selection. Then we discuss the methods used for exercise classification, repetition counting, and form analysis. Finally, we present several experiments conducted on both real-data and self-recorded exercise videos to evaluate the performance of the video.

## Problem definition

The problem can be divided into multiple sub-problems:

1. Pose Estimation: skeletonization of human
2. Exercise classification
3. Repetitive action counting: video-level or pose-level (= skeleton-level)
4. Form analysis/feedback


## Related Work

TODO: maak references af

Our main inspiration for the methods used in this project is the research done by Riccio [XXX]. To contextualize our approach, the recent advancements in computer vision driving virtual gym assistants can be categorized into pose estimation, exercise classification, repetitive action counting, and form analysis.

**Pose Estimation**
To analyze human movement, accurate and fast pose estimation is essential. Early breakthroughs in this domain were made by models like OpenPose and AlphaPose, which introduced robust multi-person pose estimation using part affinity fields and top-down bounding box detection, respectively. While highly accurate, these models can be computationally heavy. For a real-time gym assistant, low-latency processing on consumer hardware is crucial. This requirement gives Google's MediaPipe, particularly with its modern BlazePose model. By utilizing a lightweight neural network architecture, BlazePose achieves real-time 3D skeleton tracking on mobile and edge devices, forming an ideal foundation for interactive fitness tracking.

**Exercise Classification**
Once a skeleton is extracted, the next step is determining which exercise the user is performing. Recent research has shifted heavily towards deep learning on skeletal data. Graph Convolutional Neural Networks (GCNs) became a well-known technique, as they naturally model the human skeleton as a spatial-temporal graph of joints and bones. Building upon this, architectures like Graph Skeleton Transformer Networks (GSTN) have emerged, combining the structural awareness of graphs with the sequence-modeling power of Transformers, giving state-of-the-art accuracy with low latency for real-time classification of complex fitness movements.

**Repetitive Action Counting**
Counting repetitions is a core feature of any gym assistant. Initial methods utilized simple State Machines, manually defining the start and end conditions of a repetition based on specific joint angles. However, these rule-based approaches lack robustness across different body types and camera angles. This led to data-driven methods such as RepNet, which learns period length directly from video frames, and transformer-based models like TransRAC and SSTRAC, which capture long-range temporal dependencies in repetitive actions. Recent approaches emphasize efficiency and spatial-temporal skeleton dynamics; for instance, SPKDB-net leverages skeleton data for robust counting, while MSF-Mamba utilizes advanced state-space models to process long workout sequences efficiently and in real-time.

**Form Analysis and Feedback**
The most complex capability of a gym assistant is providing corrective feedback. Early systems relied heavily on logic-based thresholding, where an alert is triggered if a joint angle strays beyond a predefined limit. Modern approaches treat incorrect form as a deviation from a learned baseline, employing anomaly detection algorithms to identify mistakes dynamically without needing explicitly labeled "bad form" data. Furthermore, highly efficient architectures like MSF-Mamba are actively being extended not just for counting, but to provide rich, unified feedback, bridging the gap between simply recording a workout and actively coaching the user in real-time.

## Data

1. Introduce the dataset
2. Explain augmentation strategy
3. Motivate dataset choice and augmentation strategy

## Proposed Method

1. introducte method
2. explain why it solves the problem

## Results / Experiments

## References

[hier moet ik APA lijst van scribbr invoegen aan het einde]


