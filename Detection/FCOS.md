#### FCOS: Fully Convolutional One-Stage Object Detection
- 论文地址: https://arxiv.org/abs/1904.01355
- 作者：Zhi Tian, Chunhua Shen, Hao Chen, Tong He
- 机构：The University of Adelaide, Australia
- 代码：https://github.com/tianzhi0549/FCOS/
- 代码2：https://github.com/open-mmlab/mmdetection


#### 介绍
FCOS的网络结构和RetinaNet非常相似，不同的是多了一个Center-ness的输出。
RetinaNet中feature map上每个点对应一系列anchor，这些anchor有不同scale和aspect ration，网络的学习目标是基于这些pre-defined anchor输出score和location。FCOS直接将feature map上的每个点看成是一个sample，每个点输出对应的classification score和regression location。对应的gt计算方式也很简单，feature map上的每个点映射回原图，落在某个ground truth bounding box里面就负责预测该gt。Regression分支输出的是该点和gt四个边界的距离。

<img src='https://pic2.zhimg.com/80/v2-8917e2e47daff09aa1e4fe390f53b849_hd.jpg' />

FCOS最重要的两个地方在于FPN和Center-ness branch。目前基于anchor-based方法中，FPN都是作为backbone来使用，因为feature pyramid的存在，anchor在预定义scale的时候就根据feature stride进行了explicit划分，FPN一方面起到了特征增强的效果，另一方面也对anchor scale友好地进行了划分。<br/>
FCOS中FPN的存在，是为了解决同一个feature map点同时落在不同gt内部而造成的ambiguous sample问题。作者根据regression target将sample分配到不同level上进行预测，起到了很好的分流效果。<br/> 
在ablation study中，FPN使得AP提升了近一倍，由18.5提升到了33.8，可以说FCOS关键在于FPN结构。仔细想一下，即使是这种分流的思路，在有大量overlap的情况下，其带来的收益不会像COCO这样的数据集带来的大, ambiguous sample的问题并没有得到本质上解决。

另一个关键点在于Center-ness branch，落在gt内部比较边缘的sample有两方面缺陷，一是容易产生低质量的框，二是容易和其他gt产生overlap。为了抑制这些sample，文章提出Center-ness结构预测是中心点的置信度，最后预测框的分数由classification branch和Center-ness branch相乘得到。从实验结果来看，Center-ness提升了2.8个点，最后在ResNet-101上取得了41.0的AP，超过了CornerNet、RetinaNet等SOAT模型。


