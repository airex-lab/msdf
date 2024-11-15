# Multi-Sensor Data Fusion

## Goals: 

* Object Detection  
* Target Tracking

## Dataset: 

* [Nuscenes (website)](https://www.nuscenes.org/nuscenes),    
* [Nuscenes (arxiv)](https://arxiv.org/pdf/1903.11027)  
* [K-Radar (arxiv)](https://arxiv.org/abs/2206.08171v4)  
* [K-Radar (github)](https://github.com/kaist-avelab/k-radar)  
* [Waymo](https://waymo.com/open/)  
* [KITTI](https://www.cvlibs.net/datasets/kitti/)  
* [Lyft L5](https://www.kaggle.com/code/kool777/lyft-level5-eda-training-inference/input)

## Leaderboards: 

* [Nuscenes Detection Leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)  
* [Nuscenes Tracking Leaderboard](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)

## TODO:

* Implementation of CenterPoint, CenterNet algorithm  
* Implement Kalman filter based papers  
* Setup [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) to run multiple models which are already implemented in it  
* Literature review

# Papers

## General:

1. [Center-based 3D Object Detection and Tracking](https://github.com/tianweiy/CenterPoint)  
2. [Multimodal Virtual Point 3D Detection](https://tianweiy.github.io/mvp/)  
3. [Center Feature Fusion: Selective Multi-Sensor Fusion of Center-based Objects](https://ieeexplore.ieee.org/abstract/document/10160616)  
4. [CenterTransFuser: radar point cloud and visual information fusion for 3D object detection](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-022-00944-6)  
5. [MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection](https://arxiv.org/pdf/2408.05945)  
6. [MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving](https://arxiv.org/abs/2409.16149) \[[Code](https://github.com/megvii-research/MCTrack)\]

## Kalman Filter based:

1. [Convolutional Unscented Kalman Filter for Multi-Object Tracking with Outliers](https://www.semanticscholar.org/paper/Convolutional-Unscented-Kalman-Filter-for-Tracking-Liu-Cao/58d63ca3285648f227875762a97cbf0bf3f64af5)  
2. [Scalable Real-Time Multi Object Tracking for Automated Driving Using Intelligent Infrastructure](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10389193&tag=1)  
3. [Landmark-based RADAR SLAM for Autonomous Driving](https://www.semanticscholar.org/paper/Landmark-based-RADAR-SLAM-for-Autonomous-Driving-Ramesh-Leon/b270e4431463d6842c843310393ec040d0e946ee)  
4. [Probabilistic 3D Multi-Object Tracking for Autonomous Driving](https://www.semanticscholar.org/paper/Probabilistic-3D-Multi-Object-Tracking-for-Driving-Chiu-Prioletti/b13aabb0d92fe564e402accbf8955d75f513d9fe)

## Reinforcement Learning based:

1. [Multi-Sensor Fusion Simultaneous Localization Mapping Based on Deep Reinforcement Learning and Multi-Model Adaptive Estimation](https://www.mdpi.com/1424-8220/24/1/48)   
2. [Reinforcement Learning Based Data Fusion Method for Multi-Sensors](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9106866)  
3. [Data fusion using Bayesian theory and reinforcement learning method](http://scis.scichina.com/en/2020/170209.pdf)

## Survey Papers:

1. [Multi-sensor integrated navigation/positioning systems using data fusion: From analytics-based to learning-based approaches](https://www.sciencedirect.com/science/article/pii/S1566253523000398)  
2. [Multisensor data fusion: A review of the state-of-the-art](https://www.sciencedirect.com/science/article/pii/S1566253511000558?via%3Dihub)

## Others:

1. [Data Fusion Lexicon by Joint Directors of Laboratories](https://apps.dtic.mil/sti/pdfs/ADA529661.pdf)

# Helpful Links

* [Birds Eye View](https://multicorewareinc.com/birds-eye-view-a-primer-to-the-paradigm-shift-in-autonomous-robotics/)  
* [CV at Tesla](https://www.thinkautonomous.ai/blog/computer-vision-at-tesla/)  
* [OpenMMLab's next-generation platform for general 3D object detection](https://github.com/open-mmlab/mmdetection3d)

# Techniques

* B-Spline for time synchronisation

# NuScenes Dataset

## Detection Metrics:

### Detection Interval: **\[t-0.5, t\]**

**mAP**: Mean Average Precision (normalized area under the precision recall curve)  
**mAP** **\= (1/|C||D|)ΣΣAP**  
C \-\> Classes, D \-\> {0.5, 1, 2, 4}

**mTP**: Mean True Positive  
**mTP \= (1/|C|)ΣTP**

### True Positives:

* **ATE**: Average Translation Error (Euclidean center distance in 2D) (meters)  
* **ASE**: Average Scale Error (3D intersection over union (IOU)) after aligning orientation and translation (1 \- IOU))  
* **AOE**: Average Orientation Error (smallest yaw angle difference between prediction and ground truth) (radians)  
* **AVE**: Average Velocity Error (L2 norm of velocity differences in 2D) (m/s)  
* **AAE**: Average Attribute Error (1 \- acc) (acc: attribute classification accuracy)

**NDS**: NuScenes detection score  
**NDS \= (1/10)\[5 \* mAP \+ Σ(1 \- min(1, mTP))\]**

## Tracking Metrics:

Tracking Interval: **\[0, t\]**

**AMOTA**: Average Multi Object Tracking Accuracy  
**AMOTP**: Average Multi Object Tracking Precision  
**sMOTA**: augmented MOTA to adjust for a particula recall  
**sMOTA\_r \= max(0, 1 \- \[IDS\_r \+ FP\_r \+ FN\_r \- (1 \- r)P\]/rP)**  
**sAMOTA \= (1/|R|)ΣsMOTA\_r**

**TID**: Track initialization duration  
**LGD**: Longest gap duration  
---

# Figures

1. Joint Directors of Laboratories (JDL) Fusion Model ([source](https://www.researchgate.net/figure/Joint-Directors-of-Laboratories-JDL-Fusion-Model_fig13_252065788))

