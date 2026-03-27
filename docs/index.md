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

### Dataset
The dataset that was used for this project was the Pen action dataset. This dataset offers a variety of movements from different sports. This dataset also includes different gym exercises like squats and pushups. A convenient aspect from this dataset is that the dataset tracks the joint positions constantly. This aspect is important for training the model, because joint positions give information about the state of the exercise and whether someone has a good form. Despite that this dataset has convenient aspects, it also has a downside. All exercises from the Penn action dataset only have 1 rep. For rep counting this is not ideal, because that way the system can think that all exercises only have one rep. With data augmentation this problem is resolved, while also enlarging the dataset.
### Data augmentation
The Penn Action dataset contains relatively little data per exercise, as it includes only 2,326 videos spread across fifteen different exercises. Further the data is only limited to exercises with one rep, as discussed earlier. To expand the data to more reps, some of the videos where looped to obtain exercises with different amounts of reps. The number of reps where set to be random to obtain different rep ranges. To expand the data even further, multiple augmentations, including scaling, translating, and flipping, where applied to the data. The augmentations caused the dataset to grow with 96%, meaning that the model was trained on almost twice as much data as the original dataset.


## Methodology

### Spatial-Temporal Graph Convolutional Network

#### What Problem Does This Model Solve?

When a computer tries to understand what a human is doing, for example walking, waving or jumping, it needs a way to interpret movement. The most obvious approach might be to analyze video pixels directly, but this wastes computational effort on all kinds of visual noise that have nothing to do with the action itself. A smarter approach is to work with pose data: a representation of the human body as a set of joint positions tracked over time. Given a sequence of frames where each frame contains the 2D or 3D coordinates of joints like the head, elbows, wrists, hips and knees, the question becomes how to classify what action is being performed. The Spatial-Temporal Graph Convolutional Network (ST-GCN) was designed precisely to answer this question.

#### The Body as a Graph

The main idea of the ST-GCN is that the human body is not a grid, it is a graph. A graph is a mathematical structure made up of nodes connected by edges, and it turns out this is the most natural way to represent a skeleton. In our implementation, the graph consists of 13 nodes corresponding to the major joints: the head, shoulders, elbows, wrists, hips, knees, and ankles. The edges represent connections between joints (bones), such as the link between the left shoulder and left elbow, or between the left hip and the left knee. There is also a connection between the left and right hip, representing the pelvis as a shared anchor. Because body joints relate to each other through these physical connections, and not through their proximity on a flat grid, a graph structure captures the real relationships in a way that a regular convolutional neural network operating on a pixel grid simply cannot do.

#### Spatial Partitioning

In our implementation we structure the graph spatially, rather than treating all connections equally, the model divides each joint’s neighborhood into three distinct subsets based on the hop distance from a center node. In our case, the center is the left hip, chosen because it sits near the body’s center of mass. For any given joint, its neighbors can be joints at the same distance from the center (for example, the left and right knees are both two hops away), joints closer to the center (the hip is closer to the center than the knee), or joints further from the center (the ankle is further from the center than the knee). This three-way split, called the spatial partitioning strategy, means the model has three separate learnable transformation matrices, one for each direction of information flow. This allows the network to learn how information propagates toward the body’ core versus away from it, which turns out to be physically meaningful: the motion of the hands depends on the arms in a different way than the arms depend on the torso. Furthermore, every edge in the adjacency matrix is also paired with a learnable importance weight. These weights are initialized to one and updated during training, allowing the model to discover which joint connections are most informative for distinguishing between different actions, a push-up and squat in our case. For a push-up it might learn to look at how the joints in the arms move and for a squat it might look more at the joints in the legs.

#### Graph Convolution

The spatial graph convolution module, called unit_gcn, takes a batch of skeleton frames and propagates information across joints. For each of the three spatial subsets, it applies a learned linear projection to the joint features and then multiplies the result by the corresponding weighted adjacency matrix. The output is the sum of contributions from all three subsets, passed through batch normalization and a ReLU activation. Furthermore, a residual connections is added, the input is projected (if necessary to match channel dimensions) and added back to the output. This skip connection comes from the ResNet architecture and serves two purposes: it prevents the vanishing gradient problem in deep networks, and it ensures that each layer refines rather than replaces the representation, preserving useful information from earlier in the network.

#### Temporal Convolution

Understanding a single frame of skeleton data tells you a pose but not an action. To recognize actions, the model must also understand how poses change over time. This is handled by the temporal convolution module, called the unit_tcn, which is a standard 2D convolution applied along the time axis. It uses a kernel of size 9 in the temporal dimension and 1 in the joint dimension, meaning it looks at a window of 9 consecutive frames for each joint independently, learning patterns like the rhythm of a push-up or squat cycle.

#### Stacking Layers

The full model has seven ST-GCN blocks in sequence. The first three blocks operate at full temporal resolution and expand the channel depth from 3 to 64 channels, allowing the network to build up an internal representation of joint relationships and local motion. The fourth block introduces a stride of 2, halving the time dimension and doubling the channels to 128, the model starts to see broader motion patterns. The sixth block applies another strided downsampling, reaching 256 channels at quarter temporal resolution. By this point in the network, each feature vector encodes high-level, abstract information about how the whole body moved over the course of the clip. This progressive deepening of channels alongside temporal compression mirrors the design of standard image recognition networks, but adapted for the graph-structure of the data.

### Exercise Classification

After the layers, the model collapses the remaining spatial and temporal dimensions by taking an average across all joints and all remaining time steps. This produces a single n-dimensional vector for each sample in the batch. A single linear layer then maps this vector to the number of action classes, producing logits for each possible action. During training, these logits are passed through a cross-entropy loss function and the entire network is trained end-to-end via backpropagation, learning joint relationships, motion patterns, and combining these into action predictions.

### Form analysis
For the form analysis it is more difficult to make a data driven model. Most of the data present in gym exercise data sets only consist out of good performed exercises. This is why a rule-based form analysis is chosen. With a rule-based approach it is possible to set some thresholds for whether a good form is achieved, or that some aspects need to be improved. In this study both squads and pushups are used for form analysis. One of the most important criteria in both exercises, is that the movement is deep enough.
The approach to finding the deepness from the exercise was almost identical for both exercises. First the model checks whether a rep was made, and after that the rule base system analyses whether this rep was deep enough. For the pushup, the most important joints where the shoulders and elbows, where for squats these are the hip and knee joints. For the pushup, the moment when the shoulder is at the same height as the elbow a good rep is achieved. For the squat this is when the hip is at the same height as the knee.
For both exercises also the form of the back is analyzed. For pushups it is important to have a straight back for the whole movement. To keep track on this, a straight line was drawn from the knee position to shoulder position. If the hip was too far from this line a threshold was exceeded, which resulted in a hollow or rounded back counter going up. For the rules it is important to normalize the threshold in this specific analyzing task. For people that are longer, the threshold needs to be higher. For the squad it is important to not lean too much forward when performing. To keep track of this, the shoulders and feet should both be on the same vertical line. When this distance is too large, the person leans too much forward. Also, in this analysis, normalization is very important.

## Results / Experiments

## References

[hier moet ik APA lijst van scribbr invoegen aan het einde]
