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

