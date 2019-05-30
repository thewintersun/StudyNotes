### Libra R-CNN: Towards Balanced Learning for Object Detection
- 论文地址：https://arxiv.org/abs/1904.02701
- 作者：Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin
- 机构： 商汤，香港中文大学
- 代码地址： https://github.com/open-mmlab/mmdetection
- 介绍文章： https://mp.weixin.qq.com/s/ts4WFnuN4cHLfUh8--96Kw
- CVPR 2019 

#### 摘要
In this work, we carefully revisit the standard training practice of detectors, and find that the detection performance is often limited by the imbalance during the training process, which generally consists in three levels – sample level, feature level, and objective level. To mitigate the adverse effects caused thereby, we propose Libra R-CNN, a simple but effective framework towards balanced learning for object detection. It integrates three novel components: **IoU-balanced sampling, balanced feature pyramid, and balanced L1 loss**, respectively for reducing the imbalance at sample, feature, and objective level.

#### 三个不平衡
纵观目前主流的目标检测算法，无论 SSD、Faster R-CNN、Retinanet 这些的 detector 的设计其实都是三个步骤： 
- 选择候选区域 
- 提取特征 
- 在 muti-task loss 下收敛 

往往存在着三种层次的不平衡： 
- sample level
- feature level
- objective level 

这就对应了三个问题： 
- 采样的候选区域是否具有代表性？
- 提取出的不同 level 的特征是怎么才能真正地充分利用？
- 目前设计的损失函数能不能引导目标检测器更好地收敛？
<img src='https://mmbiz.qpic.cn/mmbiz_png/VBcD02jFhgl5ow6QwhiaWA11o5WqeMbibod69AnsVNSzU9Ticrk0zQXF2MyxK8UJ3aJicxYoXcibqJSHfQwHib3wf1icg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1' />
