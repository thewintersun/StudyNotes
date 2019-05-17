### Single-Shot Refinement Neural Network for Object Detection
论文地址：https://arxiv.org/abs/1711.06897
作者：Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei, Stan Z. Li
机构：中科院
代码：https://github.com/sfzhang15/RefineDet


#### 摘要
RefineDet consists of two inter-connected modules, namely, the anchor refinement module and the object detection module.
Specifically, the former aims to (1) filter out negative anchors to reduce search space for the classifier, and
(2) coarsely adjust the locations and sizes of anchors to provide better initialization for the subsequent regressor.

Meanwhile, we design a transfer connection block to transfer the features in the anchor refinement module to predict locations, sizes and class labels of objects in the object detection module.

#### 介绍
The one-stage approach detects objects by regular and dense sampling over locations, scales and aspect ratios. The main advantage of this is its high computational efficiency. However, its detection accuracy is usually behind that of the two-stage approach, one of the main reasons being **due to the class imbalance problem** [28].
Some recent methods in the one-stage approach aim to address the class imbalance problem, to improve the detection accuracy. 
- Kong et al. [24] use the objectness prior constraint on convolutional feature maps to significantly reduce the search space of objects.
T. Kong, F. Sun,. RON: reverse connection with objectness prior networks for object detection. In CVPR, 2017.  
- Lin et al. [28] address the class imbalance issue by reshaping the standard cross entropy loss to focus training on a sparse set of hard examples and down-weights the loss assigned to well-classified examples.
T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Doll´ar. Focal loss for dense object detection. In ICCV, 2017.
- Zhang et al. [53] design a max-out labeling mechanism to reduce false positives resulting from class imbalance.
S. Zhang, X. Zhu, Z. Lei, H. Shi, X. Wang, and S. Z. Li. S3FD: Single shot scale-invariant face detector. In ICCV,2017.

In this work, we design a novel object detection framework, called RefineDet, to inherit the merits of the two approaches (i.e., one-stage and two-stage approaches) and overcome their shortcomings. It improves the architecture of the one-stage approach, by using two inter-connected modules (see Figure 1), namely, the **anchor refinement module (ARM)** and the object detection module (ODM).

Specifically, the ARM is designed to (1) **identify and remove negative anchors** to reduce search space for the classifier, and (2) coarsely adjust the locations and sizes of anchors to provide better initialization for the subsequent regressor. The ODM takes the refined anchors as the input from the former to further improve the regression and predict multi-class labels.

In addition, we design a transfer connection block (TCB) to transfer the features3 in the ARM to predict locations, sizes, and class
labels of objects in the ODM.

#### Network Architecture
Transfer Connection Block:
