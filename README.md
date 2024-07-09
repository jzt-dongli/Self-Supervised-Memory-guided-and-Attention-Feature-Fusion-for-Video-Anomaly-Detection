# Overview
![Fig2](https://github.com/jzt-dongli/Self-Supervised-Memory-guided-and-Attention-Feature-Fusion-for-Video-Anomaly-Detection/assets/102271612/0e800623-fb2f-410d-85b7-d3418a41d6c0)
# 1.Dependencies
<pre>
python==3.7.12
torch==1.12.0+cu116
opencv-python==4.7.0.72
scikit-learn==1.0.2
</pre>
  
# 2.Code Description
This section summarizes what all the code in the project does.
## 1.AE-encoder.py
Before the features enter the memory module,
we apply an autoencoder to dimensionally compress the appearance features, 
aiming to
enhance their information density. 

First, we train
the autoencoder, optimize it using mean square
error loss (MSE loss), and process the decoder
outputs. 

Subsequently, the trained encoder
is used to compress the appearance feature a
into the compressed appearance feature.

Additionally, we employ the encoder to align the
dimensionality of appearance feature a with that
of the pose feature, ensuring compatibility with
the fusion module.

***See Section 3.3 of the paper for details.***
## 2.cluster.py & cluster_pose.py
The K-Means clustering
algorithm was employed to group the features,
and the geometric centroids of each cluster were
computed. 

These centroids were then utilized to
initialize the memory module, enabling the model
to efficiently store and recall key information. 

Initialization of the memory module is based directly
on these centroids, facilitating the ability of the
model to quickly and accurately compare and
identify similarities with known patterns upon
encountering new input data. 

To effectively characterize various normal patterns, the capacity of
the memory bank is set at M = 500 items.

***See Section 4.2 of the paper for details.***
## 3.train.py & train_pose.py
The schematic of our module is shown in Fig b). The memory bank contains M items, each recording a prototypical pattern of normal data. These items are denoted by ${p_m} \in {R^F}$($m$ = 1, ..., $M$), where each element ${p_m}$ represents a specific normal pattern. Conversely, ${q^i} \in {R^F}$ ($i$ = 1, ..., $N$) denotes the $i$th query feature, where $F$ signifies the feature dimension, and $N$ the total number of targets. We train the memory module using three types of losses: compactness, separateness, and reconstruction losses.

***See Section 3.4 of the paper for details.***
## 4.Pseudo_generators.py & Pseudo_generators_pose.py
Upon completion of training the memory module, we implement the following self-supervised scheme to generate pseudo-normal and pseudo-anomaly features,the process is shown in Fig.c).

***See Section 3.5 of the paper for details.***
## 5.padding_mask.py
In the feature fusion module, we employ a specific feature dimension unification strategy.

The four fused features are processed through an MLP network to generate the final classification output, and then the output is smoothed using a one-dimensional Gaussian filter to enhance the stability and consistency of the output.

In addition, we introduce a feature alignment strategy before the fusion module. The key is to align the three attribute features through their respective linear layers and adjust their representations during fusion to obtain consistent coding styles and distributions, thus improving the model's ability to learn feature associations.

***See Section 4.2 of the paper for details.***
## 6.attention_fusion.py
By exploring the attention mechanism, this study designs a feature fusion module that enables dynamic weighting of model components, as opposed to static parameter settings. The core principle of the attention mechanism involves assigning weights to different inputs through a learning process. Under the circumstances, each feature type—pose, velocity, appearance—can contribute varying levels of critical information depending on the situation. These three features offer diverse perspectives on the video content, enhancing both diversity and comprehensiveness in anomaly detection. While a single feature may sometimes fail or yield inaccurate results, fusing multiple features not only improves the robustness of anomaly detection but also allows for more effective identification and differentiation of normal and anomalous behaviors.

***See Section 3.6 of the paper for details.***
