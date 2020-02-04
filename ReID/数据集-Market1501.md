### Market1501

官方地址：http://www.liangzheng.com.cn/Project/project_reid.html

该数据集没有提交成绩的系统，与CVPR联合举办Workshop，目前共举办2017，2019两次。

组织者 Liang Zheng, Australian National University，联系方式：liangzheng06@gmail.com

作者整理的State-of-art 成绩LeaderBoard，==均来自论文==。

LeaderBoard地址：https://jingdongwang2017.github.io/Projects/ReID/Datasets/result_market1501.html



#### 数据集分析

 Market-1501 数据集，收自清华大学一个超市门口的==6个摄像头==，其中5个是高清摄像头，1个是低分辨率摄像头， 不同的摄像头会有视野的重叠。

![1562832263362](C:\Users\j00496872\Desktop\Notes\raw_images\1562832263362.png)

数据集包含标注好的 ==32,668 BBOX==，来自于==1,501 个对象==. 每个对象最多被6个摄像头捕获，至少被2个摄像头捕获。 该数据集有三个特征：

- First, our dataset uses the Deformable Part Model (DPM) as pedestrian detector. 
- Second, in addition to the true positive bounding boxes, we also ==provde false alarm detection results==.
- Third, each identify may have multiple images under each camera. During cross-camera search, there are multiple queries and multiple ground truths for each identity. 

The Market-1501 dataset is annotated using the following rules. For each detected bounding box to be annotated, we manually draw a ground truth bounding box that contains the pedestrian. Then, for the detected and hand-drawn bounding boxes, we ==calculate the ratio of the overlapping area to the union area==. If the ratio is larger than 50%, the DPM bounding box is marked as "good"; if the ratio is smaller than 20%, the bounding boxe is marked as "distractor"; otherwise, it is marked as "junk", meaning that this image is of zero influence to the re-identification accuracy.

**其他数据集的对比：**

| Dataset      | Market-1501 | RAiD  | CUHK03 | VIPeR | iLIDS | CUHK01 | CUHK02 | CAVIAR |
| ------------ | ----------- | ----- | ------ | ----- | ----- | ------ | ------ | ------ |
| # identities | 1,501       | 43    | 1,360  | 632   | 119   | 971    | 1,816  | 72     |
| # BBoxes     | 32,668      | 6,920 | 13,164 | 1,264 | 476   | 1,942  | 7,264  | 610    |
| #distractors | 2,793       | 0     | 0      | 0     | 0     | 0      | 0      | 0      |
| cam-per-ID   | 6           | 4     | 2      | 2     | 2     | 2      | 2      | 2      |
| DPM/Hand     | DPM         | hand  | DPM    | hand  | hand  | hand   | hand   | hand   |
| Evaluation   | mAP         | CMC   | CMC    | CMC   | CMC   | CMC    | CMC    | CMC    |

**数据集下载地址：**

[[Link 1\]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing) (Google Drive)
[[Link 2\]](http://pan.baidu.com/s/1ntIi2Op) (Baidu Disk)
[[Link 3\]](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) (A Server, Many thanks to Julian Tanke)



### 数据集简介

　　Market-1501 数据集在清华大学校园中采集，夏天拍摄，在 2015 年构建并公开。它包括由6个摄像头（其中5个高清摄像头和1个低清摄像头）拍摄到的 1501 个行人、32668 个检测到的行人矩形框。每个行人至少由2个摄像头捕获到，并且在一个摄像头中可能具有多张图像。训练集有 751 人，包含 12,936 张图像，平均每个人有 17.2 张训练数据；测试集有 750 人，包含 19,732 张图像，平均每个人有 26.3 张测试数据。3368 张查询图像的行人检测矩形框是人工绘制的，而 gallery 中的行人检测矩形框则是使用DPM检测器检测得到的。该数据集提供的固定数量的训练集和测试集均可以在single-shot或multi-shot测试设置下使用。

### 目录结构

Market-1501
　　├── bounding_box_test
　　　　　　　├── 0000_c1s1_000151_01.jpg
　　　　　　　├── 0000_c1s1_000376_03.jpg
　　　　　　　├── 0000_c1s1_001051_02.jpg
　　├── bounding_box_train
　　　　　　　├── 0002_c1s1_000451_03.jpg
　　　　　　　├── 0002_c1s1_000551_01.jpg
　　　　　　　├── 0002_c1s1_000801_01.jpg
　　├── gt_bbox
　　　　　　　├── 0001_c1s1_001051_00.jpg
　　　　　　　├── 0001_c1s1_009376_00.jpg
　　　　　　　├── 0001_c2s1_001976_00.jpg
　　├── gt_query
　　　　　　　├── 0001_c1s1_001051_00_good.mat
　　　　　　　├── 0001_c1s1_001051_00_junk.mat
　　├── query
　　　　　　　├── 0001_c1s1_001051_00.jpg
　　　　　　　├── 0001_c2s1_000301_00.jpg
　　　　　　　├── 0001_c3s1_000551_00.jpg
　　└── readme.txt

### 目录介绍

1） “bounding_box_test”——用于测试集的 750 人，包含 19,732 张图像，前缀为 0000 表示在提取这 750 人的过程中DPM检测错的图（可能与query是同一个人），-1 表示检测出来其他人的图（不在这 750 人中）

2） “bounding_box_train”——用于训练集的 751 人，包含 12,936 张图像

3） “query”——为 750 人在每个摄像头中随机选择一张图像作为query，因此一个人的query最多有 6 个，共有 3,368 张图像

4） “gt_query”——matlab格式，用于判断一个query的哪些图片是好的匹配（同一个人不同摄像头的图像）和不好的匹配（同一个人同一个摄像头的图像或非同一个人的图像）

5） “gt_bbox”——手工标注的bounding box，用于判断DPM检测的bounding box是不是一个好的box

### 命名规则

以 0001_c1s1_000151_01.jpg 为例
1） 0001 表示每个人的标签编号，从0001到1501；

2） c1 表示第一个摄像头(camera1)，共有6个摄像头；

3） s1 表示第一个录像片段(sequece1)，每个摄像机都有数个录像段；

4） 000151 表示 c1s1 的第000151帧图片，视频帧率25fps；

5） 01 表示 c1s1_001051 这一帧上的第1个检测框，由于采用DPM检测器，对于每一帧上的行人可能会框出好几个bbox。00 表示手工标注框。

### 测试协议

Cumulative Matching Characteristics (CMC) curves 是目前行人重识别领域最流行的性能评估方法。考虑一个简单的 single-gallery-shot 情形，每个数据集中的ID(gallery ID)只有一个实例. 对于每一次的识别(query), 算法将根据要查询的图像(query) 到所有gallery samples的距离从小到大排序，CMC top-k accuracy 计算如下：

```
Acc_k = 1, if top-k ranked gallery samples contain query identity
Acc_k = 0, otherwise
```

这是一个 shifted step function, 最终的CMC 曲线 (curve) 通过对所有queries的 shifted step functions取平均得到。尽管在 single-gallery-shot 情形下，CMC 有很明确的定义，但是在 multi-gallery-shot 情形下，它的定义并不明确，因为每个gallery identity 可能存在多个instances.

Market-1501中 Query 和 gallery 集可能来自相同的摄像头视角，但是对于每个query identity, 他/她的来自同一个摄像头的 gallery samples 会被排除掉。对于每个 gallery identity，他们不会只随机采样一个instance. 这意味着在计算CMC时， query 将总是匹配 gallery 中“最简单”的正样本，而不关注其他更难识别的正样本。

bounding_box_test 文件夹是 gallery 样本，bounding_box_train 文件夹是 train 样本，query 文件夹是 query 样本

由上面可以看出，在 multi-gallery-shot 情形下，CMC评估具有缺陷。因此，也使用 mAP（mean average precsion）作为评估指标。mAP可认为是PR曲线下的面积，即平均的查准率。

- [Market-1501 Evaluation Code](https://github.com/HejaBVB09/Market1501Evaluation)



### LeaderBoard分析

Baseline论文：Person re-identification: Past, Present and Future。  http://arxiv.org/abs/1610.02984

**基于监督学习方法**

| Paper Title                                                  | Year | rank1    | mAP   | Notes                                                        |
| ------------------------------------------------------------ | ---- | -------- | ----- | ------------------------------------------------------------ |
| Human Semantic   Parsing for Person Re-Identification [bibtex] | 2018 | 93.68    | 83.36 | SPReID   (combined-ft*).                                     |
|                                                              |      | 94.63    | 90.96 | SPReID   (combined-ft*) + re-ranking.                        |
| Harmonious   Attention Network for Person Re-Identification [bibtex] | 2018 | 91.2     | 75.7  | Single query.                                                |
|                                                              |      | 93.8     | 82.8  | Multiple query.                                              |
| Dual Attention Matching Network for Context-Aware Feature Sequence   Based Person Re-Identification [bibtex] | 2018 | 91.42    | 76.62 | Triplet loss +   De-Correlation loss + Cross-entropy loss with data augmentation. |
| Group Consistent Similarity Learning via Deep CRF for Person   Re-Identification[bibtex] | 2018 | 93.5     | 81.6  | Single query.                                                |
| Deep Group-Shuffling Random Walk for Person   Re-Identification [bibtex] | 2018 | 92.7     | 82.5  | Single query   with re-ranking.                              |
| Person Re-identification with Deep Similarity-Guided Graph Neural   Network[bibtex] | 2018 | 92.3     | 82.8  | Single query.                                                |
| Improving Deep   Visual Representation for Person Re-identification by Global and Local   Image-language Association[bibtex] | 2018 | 93.3     | 81.8  | Single query.                                                |
|                                                              |      | 95.3     | 87.9  | Multiple query.                                              |
| Part-Aligned   Bilinear Representations for Person Re-Identification [bibtex] | 2018 | 91.7     | 79.6  | Single query,   Inception-V1 + OpenPose.                     |
|                                                              |      | 93.4     | 89.9  | Single query,   Inception-V1 + OpenPose + re-ranking.        |
|                                                              |      | 94       | 85.2  | Multiple query,   Inception-V1 + OpenPose.                   |
|                                                              |      | 95.4     | 93.1  | Multiple query,   Inception-V1 + OpenPose + re-ranking.      |
| Mancs: A   Multi-task Attentional Network with Curriculum Sampling for Person   Re-identification [bibtex] | 2018 | 93.1     | 82.3  | Single query.                                                |
|                                                              |      | 95.4     | 87.5  | Multiple query.                                              |
| Beyond Part Models: Person Retrieval with Refined Part Pooling   (and A Strong Convolutional Baseline) [bibtex] | 2018 | 93.8     | 81.6  | Single query,   Part-based Convolutional Baseline + Refined Part Pooling. |
| A Unified   Generative Adversarial Framework for Image Generation and Person   Re-identification [bibtex] | 2018 | 92.81    | 82.67 | Single query.                                                |
|                                                              |      | 93.62    | 84.5  | Multiple query.                                              |
| CA3Net   Contextual-Attentional Attribute-Appearance Network for Person   Re-Identification [bibtex] | 2018 | 93.2     | 80    | Single query.                                                |
|                                                              |      | 94.7     | 91.5  | Single query   with re-ranking.                              |
| **Learning   Discriminative Features with Multiple Granularities for Person   Re-Identification [bibtex]** | 2018 | ==95.7== | 86.9  | Single query.                                                |
|                                                              |      | 96.6     | 94.2  | Single query   with re-ranking.                              |
|                                                              |      | 96.9     | 90.7  | Multiple query.                                              |
|                                                              |      | ==97.1== | 95.9  | Multiple query   with re-ranking.                            |
| **Local   Convolutional Neural Networks for Person Re-Identification [bibtex]** | 2018 | ==95.9== | 87.4  | Single query.                                                |
|                                                              |      | 97       | 94.7  | Single query   with re-ranking.                              |
|                                                              |      | 97.2     | 91.2  | Multiple query.                                              |
|                                                              |      | ==97.3== | 96    | Multiple query   with re-ranking.                            |
| Person   Re-identification with Hierarchical Deep Learning Feature and efficient XQDA   Metric [bibtex] | 2018 | 93.3     | 79.1  | Single query,   backbone network: ResNet.                    |
|                                                              |      | 94.3     | 90.7  | Single query,   backbone network: ResNet, with re-ranking.   |
|                                                              |      | 94.5     | 83.1  | Single query,   backbone network: DenseNet.                  |
|                                                              |      | 95.6     | 92.2  | Single query,   backbone network: DenseNet, with re-ranking. |
| Perceive Where to Focus: Learning Visibility-Aware Part-Level   Features for Partial Person Re-Identification [bibtex] | 2019 | 93       | 80.8  | Single query.                                                |
| **Densely Semantically Aligned Person   Re-Identification [bibtex]** | 2019 | ==95.8== | 87.6  | Single query.                                                |
| Re-Ranking via Metric Fusion for Object Retrieval and Person   Re-Identification[bibtex] | 2019 | 95.9     | 92.8  | Single query,   UED re-ranking.                              |
| Towards Rich Feature Discovery With Class Activation Maps   Augmentation for Person Re-Identification [bibtex] | 2019 | 94.7     | 84.5  | Single query.                                                |
| **Joint Discriminative and Generative Learning for Person   Re-Identification[bibtex]** | 2019 | 94.8     | 86    | Single query.                                                |
| AANet: Attribute   Attention Network for Person Re-Identifications [bibtex] | 2019 | 93.89    | 82.45 | Single query   without re-ranking. Use ResNet50 as backbone network. |
|                                                              |      | 93.93    | 83.41 | Single query   without re-ranking. Use ResNet152 as backbone network. |
|                                                              |      | 95.1     | 92.38 | Single query   with re-ranking. Use ResNet152 as backbone network. |
| **Pyramidal Person Re-IDentification via Multi-Loss Dynamic   Training [bibtex]** | 2019 | 95.7     | 88.2  | Single query   without re-ranking.                           |
| Interaction-And-Aggregation Network for Person   Re-Identification [bibtex] | 2019 | 94.4     | 83.1  | Single query   without re-ranking.                           |

