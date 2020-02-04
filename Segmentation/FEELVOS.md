### FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation

论文地址：https://arxiv.org/abs/1902.09513v2

作者：  Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, Liang-Chieh Chen

机构： RWTH Aachen University, Google Inc.

发表：CVPR 2019

代码地址：https://github.com/tensorflow/models/tree/master/research/feelvos



#### 摘要

最近许多成功的视频对象分割(VOS)方法过于复杂，严重依赖于对第一帧的微调，且或速度较慢，因此实际应用有限。在这项工作中，我们提出FEELVOS一种不依赖于微调的简单、快速的方法。

分割视频过程中，FEELVOS对每一帧都使用 semantic pixel-wise embedding，并结合全局和局部匹配机制，将信息从第一帧和前一帧传输到当前帧。与之前的工作相比，我们的embedding 只是作为卷积网络的内部指导。Our novel dynamic segmentation head allows us to train the network, including the embedding, ==end-to-end for the multiple object segmentation== task with a ==cross entropy loss==. 

我们在不fine-tuning的情况下实现了视频对象分割的 state of the art，在DAVIS 2017验证集上J&F测量值为71.5%。



#### 介绍

视频对象分割(VOS)是计算机视觉中的一项基本任务，在视频编辑、机器人和自动驾驶汽车等领域有重要应用。文章关注重点在半监督VOS，也就是会给出第一帧的对象的Mask Label，然后推出其他帧中该对象的Mask。

作者设计目标Simple, Fast, End-to-End, Strong， 灵感来自于论文Pixel-Wise Metric Learning (PML) [6] . 

> Y. Chen, J. Pont-Tuset, A. Montes, and L. Van Gool. Blazingly fast video object segmentation with pixel-wise metric learning. In CVPR, 2018. 

PML ==learns a pixel-wise embedding using a triplet loss== and at test time ==assigns a label to each pixel by nearest neighbor matching in pixel space to the first frame==. PML 的实现是simple and fast 的，但不是End-to-End的，并且会产生 noisy segmentations due to the hard assignments via nearest neighbor matching.

FEELVOS 采用 learned embedding和最邻近匹配,但我们用这个机制作为卷积网络的内部指导,而不是用它来做最终的分割决定。这使得我们可以用端到端的方式embedding，通过对分割输出使用标准的交叉熵损失。

通过使用最邻近匹配只作为一个软线索, 网络可以从部分不正确的nearest neighbor assignments中恢复过来, 并仍然产生准确的分割。在没有 fine-tuning 情况下,我们实现了多目标分割的 state of the art 。

![1569059249894](C:\Users\j00496872\Desktop\Notes\raw_images\1569059249894.png)

图1. FEELVOS 方法的概述。

对当前帧的图像进行分割（采用 DeepLabV3+作主干网络，去掉输出层，采用Stride 4）, 提取它的backbone 特征和 pixel-wise embedding 向量。

然后这pixel-wise embedding 向量 globally 与第一帧进行匹配 并 locally与前一帧进行匹配，分别产生一个global distance map 和一个local distance map。

作者将global distance map, local distance map, 当前帧的backbone features，前一帧的预测结果，这四种信息一起（at stride 4），送到 a dynamic segmentation head, 输出最后的分割结果。 



#### **相关工作**

**1. Video Object Segmentation with First-Frame Fine-tuning**

- OSVOS [1] uses a convolutional network,  pre-trained for foreground-background segmentation, and fine-tunes it on the first-frame ground truth of the target video at test time.  OnAVOS [35, 34] and OSVOS-S [27] extend OSVOS by an on-line adaptation mechanism, and by semantic information from an instance segmentation network, respectively. 

- Another approach is to learn to ==propagate the segmentation mask from one frame to the next using optical flow== as done by MaskTrack [28]. This approach is extended by LucidTracker [20] which introduces an elaborate data augmentation mechanism. 

- Hu et al. [15] propose ==a motion-guided cascaded refinement network== which works on a coarse segmentation from an active contour model. 

- MaskRNN [16] uses a recurrent neural network to fuse the output of two deep networks. 

- ==Location-sensitive embeddings== used to refine an initial foreground prediction are explored in LSE [9]. 
- MoNet [38] ==exploits optical flow motion cues== by feature alignment and ==a distance transform layer==.
- Using reinforcement learning to ==estimate a region of interest to be segmented== is explored by Han et al. [13]. 
- DyeNet [22] uses a deep recurrent network which ==combines a temporal propagation and a re-identification module==. 

- PReMVOS [26, 24, 25] combines ==four different neural networks together== with extensive fine-tuning and a merging algorithm and won the 2018 DAVIS Challenge [2] and also the 2018 YouTube-VOS challenge [39].

**2. Video Object Segmentation without First-Frame Fine-tuning.**

- OSMN[40] combines a segmentation network with a modulator, which manipulates intermediate layers of the segmentation network without requiring fine-tuning. 
- FAVOS [7] uses a part-based tracking method to obtain bounding boxes for object parts and then produces segmentation masks using a region-of-interest based segmentation network. 
- The main inspiration of the proposed FEELVOS approach is PML [6], which uses a pixel-wise embedding learned with a triplet loss together with a nearest neighbor classifier.
- VideoMatch[17] ==uses a soft matching layer== which is very similar to PML and considers for each pixel in the current frame the closest k nearest neighbors to each pixel in the first frame in a learned embedding space. Unlike PML, it directly optimizes the resulting segmentation instead of using a triplet loss. However, the final segmentation result is still directly derived from the matches in the embedding space which makes it hard to recover from incorrect matches.
- RGMP [37] uses a Siamese encoder with two shared streams. The first stream encodes the video frame to be segmented together with the estimated segmentation mask of the previous frame. The second stream encodes the first frame of the video together with its given ground truth segmentation mask. The features of both streams are then concatenated and combined by a global convolution block and multiple refinement modules to produce the final segmentation mask. 

 作者的主要灵感来自于PML， 和 VideoMatch 和 RGMP 有很多共通之处。



#### 方法

提出的体系结构以==DeepLabv3+(去掉输出层)为骨干网络==，步长为4提取特征。

**Semantic Embedding**

Embedding的想法来源于：属于相同对象实例的像素(在同一帧或不同帧中)，在embedding space 中是接近的， 而属于不同物体的像素将会很远。

For each pixel p, we extract a semantic embedding vector $e_p$ in the learned embeddings pace. Similar to Fathi et al. [12], 定义两个像素 p 和 q 对应的 embedding vectors $e_p$ and $e_q$的距离通过公式: 

![1569401480454](C:\Users\j00496872\Desktop\Notes\raw_images\1569401480454.png)

**Global Matching**

让$P_t$ 表示时间t上所有像素的集合(stride等于4)，  和$P_{t,o}$ 表示时间t上属于对象o的所有像素的集合。

计算全局匹配距离地图 $G_{t,o}(p)$ ，对于每个ground truth 对象o， 当前时间t的视频帧的每个像素p 与 该对象在第一帧中的像素集合$P_{1,o}$的最邻近的距离：

![1569401606278](C:\Users\j00496872\Desktop\Notes\raw_images\1569401606278.png)

![1569403094135](C:\Users\j00496872\Desktop\Notes\raw_images\1569403094135.png)

图2。Global and local matching。对于给定的对象(在本例中为duck)，global matching 将当前帧的embedding 向量与属于该对象的第一帧的embedding 向量匹配，并生成distance map。深色表示距离较小。==注意，全局distance map是有噪声的，并且在水中包含假阳性。==Local matching用于将当前帧embedding 匹配到属于对象的前一帧的embedding 。对于local matching，一个像素的匹配只允许在它周围的局部窗口中进行。

**Local Previous Frame Matching**

与前一帧的Embedding的distance map 计算方法：

![1569403896651](C:\Users\j00496872\Desktop\Notes\raw_images\1569403896651.png)

$P_{t-1,o}$ 是t-1时间的预测结果，所以对于对象o, 它可能是不存在的，这时候，就设置距离为1.

对于第一帧的Global matching，我们计算当前帧所有像素和第一帧的Object o的距离，但是对于前一帧，我们没有必要这么计算，因为当前帧和前一帧的移动距离并不大，所以只计算前一帧的像素p的邻近像素点集合的距离：

![1569404404448](C:\Users\j00496872\Desktop\Notes\raw_images\1569404404448.png)

邻近点的计算：

Inspired by FlowNet [11], for pixel p of frame t we ==only consider pixels q of frame t - 1 in a local neighborhood of p when searching for a nearest neighbor==. For a given window size k, we define the neighborhood ==N(p) as the set of pixels (regardless of which frame they are coming from) which are at most k pixels away from p in both x and y direction==. This means that N(p) usually comprises (2*k + 1)^2 elements in the same frame where p is coming from, and fewer elements close to the image boundaries.

**Previous Frame Predictions**

使用前一帧的分割结果作为一种特征，来进行当前帧的分割。

**Dynamic Segmentation Head**

为了系统有效地处理多个对象，我们提出了一个动态分割头，该分割头对每个具有共享权值的对象进行一次动态实例化。

![1569467710944](C:\Users\j00496872\Desktop\Notes\raw_images\1569467710944.png)

图3. 动态分割头用于系统地处理多个对象。轻量级分割头为视频中的每个对象动态实例化一次，并为每个对象logits 生成的一维 feature map 。然后将每个对象的 logits 堆叠在一起并应用 softmax 。采用标准交叉熵损失训练动态分割头。

**Training procedure**

我们的训练程序特意设置得很简单。对于每个训练步骤，我们首先随机选择一小批视频。==对于每个视频，我们随机选择三帧:一帧作为参考帧，即，它起视频第一帧的作用，相邻的两帧中，第一帧作为前一帧，第二帧作为当前要分割的帧。== 我们只对当前帧应用损失函数。在训练过程中，我们使用前一帧的ground truth进行局部匹配，并使用它来定义前一帧的预测，方法是对正确的对象将其设置为1，对每个像素的所有其他对象将其设置为0。

**Inference**

给出了一个带有第一帧ground truth的测试视频，首先提取第一帧的embedding vectors。然后，我们逐帧遍历视频，计算当前帧的嵌入向量，对第一帧应用全局匹配，对前一帧应用局部匹配，对每个对象run动态分割头，并应用pixel-wise argmax生成最终分割结果。



#### 实验

作为我们网络的骨干，我们使用最新的DeepLabv3+架构[5]，该架构基于Xception-65[8,33]架构，采用深度可分卷积 depth-wise separable convolutions [14]，batch normalization[18]， Atrous空间金字塔池 Atrous Spatial Pyramid Pooling[3,4]，以及一个Stride等于4的解码器模块。

We extract embedding vectors of dimension 100. 

![1569472478575](C:\Users\j00496872\Desktop\Notes\raw_images\1569472478575.png)

![1569472579318](C:\Users\j00496872\Desktop\Notes\raw_images\1569472579318.png)

**Ablation Study**

![1569472690463](C:\Users\j00496872\Desktop\Notes\raw_images\1569472690463.png)

实验样例

![1569472857642](C:\Users\j00496872\Desktop\Notes\raw_images\1569472857642.png)

