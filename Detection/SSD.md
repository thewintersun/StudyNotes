SSD: Single Shot MultiBox Detector
- 论文地址：https://arxiv.org/abs/1512.02325
- 作者：Wei Liu, Dragomir Anguelov, Dumitru Erhan
- 机构：Google
- ECCV 2016

### 摘要
Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales
per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to
the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle
objects of various sizes.


### 介绍
Our improvements include：
- using a small convolutional filter to predict object categories and offsets in bounding box locations, 
- using separate predictors (filters) for different aspect ratio detections, 
- and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales. 
With these modifications—especially using multiple layers for prediction at different scales—we can achieve high-accuracy using relatively low resolution input, further increasing detection speed.


