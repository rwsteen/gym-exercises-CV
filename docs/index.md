# Virtual Gym Assistance using Deep Learning

*By: Rogier van Tienen, Rayan Salmi & Ricardo Steen*

*From: Delft University of Technology*

*Date: 7th of April 2026*

*Group: 1 (The three R's)*

---

## Introduction

In the last years, the popularity of the gym has been increasing. Most of these gyms have employees working in the gym to help people with exercises, but there are also a large number of gyms where no employee is present. This can cause difficulties for people who have just started to work out and have questions about certain gym exercises they are performing. To come with a solution, this blog proposes a virtual gym assistant. With new computer vision techniques like more accurate pose estimation methods, it becomes easier to track human movement. The proposed virtual gym assistant uses these techniques to classify the type of exercise, count the repetitions, and give feedback on how the exercise is performed and whether the form was correct. 

The problem can be divided into multiple sub-problems:

1. Pose Estimation: skeletonization of human
2. Exercise classification
3. Repetitive action counting: video-level or pose-level (= skeleton-level)
4. Form analysis/feedback

In this blog, we first discuss related work to get an idea of the models that are currently used for the aforementioned computer vision tasks. Next, we introduce the dataset used in this study and explain the motivation behind its selection. Then we discuss the methods used for exercise classification, repetition counting, and form feedback. Finally, we present several experiments conducted on both test-data and unseen self-recorded exercise videos to evaluate the performance of the video.

## Related Work
Our main inspiration for this project is the research done by Riccio[14]. In his paper, Riccio showed a computer vision pipeline that was able to classify exercises and count repetitions. To improve upon Riccio's work, we decided to include form analysis and feedback. And for further literature review, we looked at recent advancements in computer vision driving virtual gym assistants. These advancements can be categorized into pose estimation, exercise classification, repetitive action counting (RAC), and form analysis.

**Pose Estimation:** 
To succesfully analyze human movement, accurate and fast pose estimation is essential. Early breakthroughs in this domain were made by models like OpenPose and AlphaPose, which introduced robust multi-person pose estimation using part affinity fields and top-down bounding box detection [3][6]. While highly accurate, these models can be computationally heavy. For a real-time gym assistant, low-latency processing on consumer hardware is crucial. This requirement leads to MediaPipe, particularly with its modern BlazePose model [13][2]. By utilizing a lightweight neural network architecture, BlazePose achieves real-time 3D skeleton tracking on mobile and edge devices, forming an ideal foundation for a virtual gym assistant.

**Exercise Classification:**
Once a skeleton is extracted, the next step is determining which exercise the user is performing. Recent research has shifted heavily towards deep learning on skeletal data. Graph Convolutional Neural Networks (GCNs) became a well-known technique, as they naturally model the human skeleton as a spatial-temporal graph of joints and bones [15]. Building upon this, architectures like Graph Skeleton Transformer Networks (GSTN) have emerged, combining the structural awareness of graphs with the sequence-modeling power of Transformers, giving state-of-the-art accuracy with low latency for real-time classification of fitness movements[8].

**Repetitive Action Counting (RAC):**
Counting repetitions is a core feature of any gym assistant. Initial methods utilized simple State Machines, manually defining the start and end conditions of a repetition based on specific joint angles. However, these rule-based approaches lack robustness across different body types and camera angles. This led to data-driven methods such as RepNet, which learns period length directly from video frames, and transformer-based models like TransRAC and SSTRAC, which capture long-range temporal dependencies in repetitive actions [5][7][12]. Recent approaches emphasize efficiency and spatial-temporal skeleton dynamics; for instance, SPKDB-net[1] leverages salient-part pose keypoints for robust counting, while MSF-Mamba is considered state-of-the-art for RAC, as it combines linear state-space models with a motion-aware state fusion mechanism to detect subtle temporal patterns and repetitions efficiently and in real-time [11].

**Form Analysis and Feedback:**
The most complex capability of a gym assistant is providing corrective feedback. Early systems relied heavily on logic-based algorithms, where an alert is triggered if a joint angle strays beyond a predefined limit [4]. Modern approaches treat incorrect form as a deviation from a learned baseline, employing anomaly detection algorithms to identify mistakes without needing explicitly labeled "bad form" data [9][10]. Furthermore, highly efficient methods like MSF-Mamba are actively being extended not just for counting, but to provide rich feedback, bridging the gap between simply tracking a workout and actively coaching the user in real-time.

## Data

### Dataset
The dataset used was the Penn action dataset[16]. This dataset offers a variety of movements from different sports. This dataset also includes different gym exercises like squats and push-ups. A convenient aspect of this dataset is that it tracks the joint positions constantly. This aspect is important for training the model, because joint positions give information about the state of the exercise and whether someone has good form. Despite the convient aspects of the dataset, it also has a downside. All exercises from the Penn action dataset only have one rep. For rep counting this is not ideal, because the system would treat every exercise as if it had only one rep. With data augmentation this problem is resolved, while also enlarging the dataset.
### Data augmentation
The Penn Action dataset contains relatively little data per exercise, with only 2,326 videos across fifteen different exercises. In our project, we focus on squats and push-ups, which together account for approximately 500 videos. To generate exercises with multiple repetitions, some videos were looped to create sequences with varying numbers of reps. Additionally, the data was augmented through techniques such as scaling, translation, and flipping. These augmentations increased the dataset by 96%, effectively nearly doubling the amount of training data available to the model.


## Methodology

### Spatial-Temporal Graph Convolutional Network

#### What Problem Does This Model Solve?

Recognizing human exercises directly from raw video frames is computationally expensive and sensitive to visual noise. A more efficient approach is to track body joints over time and classify actions based on this skeletal movement. The Spatial-Temporal Graph Convolutional Network (ST-GCN)[15] is designed for this task.

#### The Body as a Graph

ST-GCN treats the human skeleton as a graph, where joints are nodes and bones are edges. This allows the model to capture natural relationships between body parts, like how elbows move relative to shoulders, better than standard grid-based methods.

#### Spatial Partitioning

Instead of treating all neighboring joints equally, ST-GCN divides them into three groups based on distance from a central node (the left hip): same distance, closer, or further away. Each group has its own learnable weights, allowing the model to understand how motion flows through the body. Additional learnable edge weights help the model focus on the most relevant joints for different actions.

#### How ST-GCN Understands Movement

The model divides neighboring joints into groups based on their distance from a central joint and assigns learnable weights to each group. This helps it understand how motion flows through the body. Over time, temporal convolutions analyze sequences of frames, learning patterns like repetitions and rhythm in exercises.

#### Layered Structure

ST-GCN stacks multiple blocks that combine spatial and temporal processing. Early layers capture short-term, local motion, while deeper layers learn longer-term, more abstract movement patterns. This design balances accuracy with computational efficiency, making it suitable for real-time gym assistance.

![My SVG](./stgcn_model.svg)
*Figure 1: Pipeline of the ST-GCN model for exercise classification. Input skeleton sequences are normalized before passing through four backbone blocks combining spatial and temporal processing. Features are then pooled and classified to predict the exercise.*

### Exercise Classification

After processing skeletal sequences, the model summarizes the movement of all joints over time and predicts which exercise is being performed. This allows the system to recognize exercises like squats or push-ups accurately in real time, providing the foundation for repetition counting and form feedback.

### Repetition Counting
We implemented two approaches to count repetitions. The first is heuristic-based, relying on joint angles to detect transitions between phases (e.g., “up” and “down”). Each time the movement passes through these phases, a rep is counted. The second is learning-based, where the model predicts a continuous movement phase from 0 to 1. By tracking changes in this phase over time, repetitions are counted more robustly, even with different body types and camera angles. The learning-based approach generally provides higher accuracy and smoother counting.

### Form analysis
Detecting incorrect form is challenging because most datasets contain only correctly performed exercises. One possible approach would be to learn deviations from the distribution of good exercises using anomaly detection. While this can identify that something is “off,” it does not specify what exactly is wrong in the execution. To provide actionable feedback, we instead use a simple rule-based approach.

For squats and push-ups, the system checks whether each repetition is deep enough and whether the back posture is correct. For push-ups, a proper rep is when the shoulders reach the same height as the elbows, with a straight back throughout. For squats, the hips should reach knee height without leaning too far forward. Thresholds are normalized for body size to ensure fairness across users. This method allows the assistant to give specific, understandable feedback, helping users perform exercises safely and effectively.

## Results / Experiments
### Experimental Questions
To evaluate the virtual gym assistant, we ask the following questions:

1. Exercise classification: Can the system correctly identify exercises like push-ups and squats across different participants, filming angles, and lighting conditions?
2. Repetition counting: Can the system reliably count repetitions, even when exercise speed and depth vary?
3. Form analysis: Can the system detect shallow repetitions or incorrect postures, and does the learning-based approach improve robustness compared to the heuristic-based approach?

These questions guide the experiments and allow us to measure whether the virtual gym assistant can provide accurate, safe, and real-time feedback to users.

### Experiments
To test the model on robustness multiple experiments were conducted with video’s that were not in the dataset. These videos were self-created and included both squats and push-up of different rep ranges and were filmed from different angles. To create variety in the test data, different locations were tested, with three participants in total. The test data consisted out of 20 video’s for push-ups and 20 video’s for squats. Each video was validated separately with an accuracy for exercise prediction and an absolute error for rep counting, these metrics were automatically conducted by the application. Furthermore, the validation per exercise was done to obtain visual feedback from each video. The visualization is also obtained by the application, this visualization shows the video with the joint positions that were recorded, including the rep count and exercise prediction. The total result is the average classification accuracy and average absolute repetition error across push-ups, squats, and all exercises combined.

As an extension of the experiments, exercise form was also analyzed. Consisting of the amount of shallow reps and the actual form. The amount of shallow reps is computed in the same way as the total amount of reps, the only difference is that a threshold is set for the deepness of the exercise. The form was obtained in the same way for both the heuristic-based and learning-based approaches. Differences arise because the analysis depends on the exercise phase and predicted exercise, which vary between the approaches. Receiving results for the form analysis is harder to achieve than with the exercise prediction and rep count. The form analysis checks every frame whether a good form is achieved or whether the form is incorrect. Labeling each frame will take a lot of time, that is why a different approach is chosen. For this experiment, the labels include the number of correct repetitions, as well as the number of repetitions performed with specific form deviations (hollow back, rounded back, or forward lean).For each form type, the number of frames in which it occurs is counted and summed over the entire video. These counts are then converted into fractions representing the proportion of each form type. Each fraction is multiplied by the total number of actual repetitions to estimate the number of repetitions per form type. These estimates are then compared to the labeled counts, and the absolute error is computed for each form type.

### results
The results for the exercise classification and absolute rep count error are shown in the tables below. Overall the learning based approach scores best with a higher accuracy for exercise prediction and a lower absolute rep count error. The primary challenge for the heuristic-based approach was camera placement, because the angles between joints can vary depending on the placement of the camera. Since the heuristic model relies heavily on joint angles, its performance is more sensitive to variations in camera perspective. In contrast, the learning-based model can more effectively learn patterns that generalize across different angles. Additional factors that influenced model performance for both models included lighting conditions, extremely bad posture, outside frame joint positioning and abrupt video endings, where recordings stopped immediately after the final repetition.

#### Heuristic-based

| Exercise  | Mean accuracy exercise prediction | Mean absolute error reps |
|----------|------------------------------|---------------------|
| Push-up   | 0.7315                       | 1.0500              |
| Squat    | 1.0000                       | 3.0000              |
| Combined | 0.8658                       | 2.0250              |

#### Learning-based

| Exercise  | Mean accuracy exercise prediction | Mean absolute error reps |
|----------|------------------------------|---------------------|
| Push-up   | 1.0000                       | 0.9000              |
| Squat    | 0.9251                       | 0.8000              |
| Combined | 0.9625                       | 0.8500              |

The results of the form analysis are presented in the tables below. A notable finding is the large difference in push-up form performance between the heuristic-based and learning-based approaches. The heuristic-based method relies heavily on joint positions and angles between joints at each frame, making it sensitive to occlusions, camera angle, and minor detection errors. This lack of robustness likely contributes to the higher absolute errors observed. In contrast, the learning-based approach does not require precise joint positions at every frame, as it learns motion patterns and relationships from sequences of poses, making it more resilient to missing or noisy joint data. For squats, the difference in mean absolute error between the two methods was smaller, indicating that both approaches are relatively consistent for this exercise. These results highlight the importance of robust exercise classification and motion modeling for reliable form analysis, especially in exercises with complex joint movements.

#### Push-up: Absolute Error per Form Type

| Method           | Mean absolute error shallow rep | Mean absolute error good form | Mean absolute error rounded back | Mean absolute error hollow back |
|------------------|----------------------------|--------------------------|------------------------------|----------------------------|
| Heuristic-based  | 1.5000                     | 2.5658                   | 0.7938                       | 1.3645                     |
| Learning-based   | 0.8500                     | 0.5580                   | 0.3126                       | 0.2530                     |

#### Squat: Absolute Error per Form Type

| Method           | Mean absolute error shallow rep | Mean absolute error good form | Mean absolute error forward lean |
|------------------|----------------------------|--------------------------|------------------------------|
| Heuristic-based  | 0.4500                     | 0.2643                   | 0.2643                       |
| Learning-based   | 1.5500                     | 0.3124                   | 0.1282                       |



## References

1. Jinying Wu, Jun Li, Qiming Li, SPKDB-Net: A Salient-Part Pose Keypoints-Based Dual-Branch Network for repetitive action counting, Computer Vision and Image Understanding, Volume 259, 2025, 104434, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2025.104434.

2. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020, 17 juni). BlazePose: On-device Real-time Body Pose tracking. arXiv.org. https://arxiv.org/abs/2006.10204 

3. Cao, Z., Hidalgo, G., Simon, T., Wei, S., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D pose Estimation using part affinity fields. IEEE Transactions On Pattern Analysis And Machine Intelligence, 43(1), 172–186. https://doi.org/10.1109/tpami.2019.2929257 

4. Chen, S., & Yang, R. R. (2020, 21 juni). Pose Trainer: Correcting Exercise Posture using Pose Estimation. arXiv.org. https://arxiv.org/abs/2006.11718 

5. Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A. (2020). Counting Out Time: Class Agnostic Video Repetition Counting in the Wild. IEEE/CVF Conference On Computer Vision And Pattern Recognition (CVPR), 10384–10393. https://doi.org/10.1109/cvpr42600.2020.01040 

6. Fang, H., Li, J., Tang, H., Xu, C., Zhu, H., Xiu, Y., Li, Y., & Lu, C. (2022). AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time. IEEE Transactions On Pattern Analysis And Machine Intelligence, 45(6), 7157–7173. https://doi.org/10.1109/tpami.2022.3222784 

7. Hu, H., Dong, S., Zhao, Y., Lian, D., Li, Z., & Gao, S. (2022). TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting. 2022 IEEE/CVF Conference On Computer Vision And Pattern Recognition (CVPR), 18991–19000. https://doi.org/10.1109/cvpr52688.2022.01843 

8. Jiang, Y., Sun, Z., Yu, S., Wang, S., & Song, Y. (2022). A Graph Skeleton Transformer Network for Action Recognition. Symmetry, 14(8), 1547. https://doi.org/10.3390/sym14081547 

9. Kowsar, Y., Moshtaghi, M., Velloso, E., Kulik, L., & Leckie, C. (2016). Detecting unseen anomalies in weight training exercises. In OzCHI ’16: Proceedings of the 28th Australian Conference on Computer-Human Interaction (pp. 517–526). https://doi.org/10.1145/3010915.3010941

10. LAZIER: A Virtual Fitness Coach Based on AI Technology. (2022, 23 september). IEEE Conference Publication IEEE Xplore. https://ieeexplore.ieee.org/document/9927664 

11. Li, D., Shao, J., Xing, B., Gao, R., Wen, B., Kälviäinen, H., & Liu, X. (2026). MSF-Mamba: Motion-aware State Fusion Mamba for Efficient Micro-Gesture Recognition. IEEE Transactions On Multimedia, 1–12. https://doi.org/10.1109/tmm.2026.3668511 

12. Lim, J., Kang, D., Ryu, K., & Hong, J. H. (2025). SSTRAC: Skeleton-Based Dual-Stream Spatio-Temporal Transformer for Repetitive Action Counting in Videos. IEEE Access, 13, 184046–184058. https://doi.org/10.1109/access.2025.3624029 

13. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C., Yong, M., Lee, J., Chang, W., Hua, W., Georg, M., & Grundmann, M. (2019, 1 januari). MediaPipe: A Framework for Perceiving and Processing Reality. https://research.google/pubs/pub48292/ 

14. Riccio, R. (2024). Real-Time fitness exercise classification and counting from video frames. arXiv Preprint, arXiv:2411.11548.

15. Yan, S., Xiong, Y., & Lin, D. (2018, April). Spatial temporal graph convolutional networks for skeleton-based action recognition. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).
    
16. Weiyu Zhang, Menglong Zhu and Konstantinos Derpanis,  "From 
Actemes to Action: A Strongly-supervised Representation for
Detailed Action Understanding" International Conference on 
Computer Vision (ICCV). Dec 2013.
