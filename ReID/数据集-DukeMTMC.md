#### DukeMTMC-reID

官网地址：http://vision.cs.duke.edu/DukeMTMC/ （暂时无法打开）

数据下载地址：http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip 

Github评价工具：https://github.com/naiq/Duke_evaluation （matlab 代码）

数据下载地址：[GoogleDriver](https://drive.google.com/open?id=0B0VOCNYh8HeRSDRwczZIT0lZTG8) or ([BaiduYun](https://pan.baidu.com/s/1cIOYOu) password:48z4).

数据集介绍论文： http://openaccess.thecvf.com/content_cvpr_2017_workshops/w17/papers/Gou_DukeMTMC4ReID_A_Large-Scale_CVPR_2017_paper.pdf



#### 数据集简介

DukeMTMC-reID 为 DukeMTMC数据集的行人重识别子集。DukeMTMC 为行人跟踪数据集。

原始数据集包含了85分钟的高分辨率视频，采集自8个不同的摄像头。并且提供了人工标注的bounding box.

我们从视频中每120帧采样一张图像，得到了 ==36,411张图像==。一共有==1,404个人==出现在大于两个摄像头下，有==408个人==只出现在一个摄像头下。所以我们随机采样了 702 个人作为训练集，702个人作为测试集。在测试集中，我们采样了每个ID的每个摄像头下的一张照片作为查询图像（query）。剩下的图像加入测试的搜索库（gallery），并且将之前的 408人作为干扰项，也加到 gallery中。



#### 目录结构

DukeMTMC-reID
　　├── bounding_box_test
　　　　　　　├── 0002_c1_f0044158.jpg
　　　　　　　├── 3761_c6_f0183709.jpg
　　　　　　　├── 7139_c2_f0160815.jpg
　　├── bounding_box_train
　　　　　　　├── 0001_c2_f0046182.jpg
　　　　　　　├── 0008_c3_f0026318.jpg
　　　　　　　├── 7140_c4_f0175988.jpg
　　├── query
　　　　　　　├── 0005_c2_f0046985.jpg
　　　　　　　├── 0023_c4_f0031504.jpg
　　　　　　　├── 7139_c2_f0160575.jpg
　　└── CITATION_DukeMTMC.txt
　　└── CITATION_DukeMTMC-reID.txt
　　└── LICENSE_DukeMTMC.txt
　　└── LICENSE_DukeMTMC-reID.txt
　　└── README.md

从视频中每 120 帧采样一张图像，得到了 36,411 张图像。一共有 1,404 个人出现在大于两个摄像头下，有 408 个人 (distractor ID) 只出现在一个摄像头下。
1） “bounding_box_test”——用于测试集的 702 人，包含 ==17,661 张图像==（随机采样，702 ID + 408 distractor ID）
2） “bounding_box_train”——用于训练集的 702 人，包含 ==16,522 张图像==（随机采样）

3） “query”——为测试集中的 702 人在每个摄像头中随机选择一张图像作为 query，共有 ==2,228 张图像==



**图像命名规则为**

"0005_c2_f0046985.jpg", 

- "0005" 代表行人的身份. 
- "c2"代表这张图像来自第二个摄像头. 
- "f0046985" 代表来自摄像头2的 第46985帧.

![1562837803381](C:\Users\j00496872\Desktop\Notes\raw_images\1562837803381.png)

另外，DukeMTMC-reID还提供了==23种属性数据标注== DukeMTMC-attribute供下载。https://github.com/vana77/DukeMTMC-attribute



#### LeaderBoard 分析

作者提供的Baseline成绩：

![1562838682157](C:\Users\j00496872\Desktop\Notes\raw_images\1562838682157.png)

官网打不开，没有找到19年的LeaderBoard，有18年整理的的：

 https://github.com/yang502/DukeMTMC-reID_evaluation/tree/master/State-of-the-art



| Methods                               | Rank@1 | mAP    | Reference                                                    |
| ------------------------------------- | ------ | ------ | ------------------------------------------------------------ |
| ATWL(2-stream)                        | 79.80% | 63.40% | "[Features for Multi-Target Multi-Camera Tracking and Re-Identification](https://arxiv.org/abs/1803.10859)", Ergys Ristani and Carlo Tomasi, CVPR 2018 |
| Mid-level Representation              | 80.43% | 63.88% | "[The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching](https://arxiv.org/abs/1711.08106)", Qian Yu, Xiaobin Chang, Yi-Zhe Song, Tao Xiang, Timothy M. Hospedales, arXiv:1711.08106 |
| HA-CNN                                | 80.5%  | 63.8%  | "[Harmonious Attention Network for Person Re-Identification](https://arxiv.org/abs/1802.08122)", Li Wei, Xiatian Zhu, and Shaogang Gong, CVPR 2018 |
| Deep-Person                           | 80.90% | 64.80% | "[Deep-Person: Learning Discriminative Deep Features for Person Re-Identification](https://arxiv.org/abs/1711.10658)", Xiang Bai, Mingkun Yang, Tengteng Huang, Zhiyong Dou, Rui Yu, Yongchao Xu, arXiv:1711.10658 |
| MLFN                                  | 81.2%  | 62.8%  | "[Multi-Level Factorisation Net for Person Re-Identification](https://arxiv.org/abs/1803.09132)" Xiaobin Chang, Timothy M. Hospedales, and Tao Xiang, CVPR 2018. |
| DuATM (Dense-121)                     | 81.82% | 64.58% | "[Dual Attention Matching Network for Context-Aware Feature Sequence based Person Re-Identification](https://arxiv.org/abs/1803.09937)", Jianlou Si, Honggang Zhang, Chun-Guang Li, Jason Kuen, Xiangfei Kong, Alex C. Kot, Gang Wang, CVPR 2018 |
| PCB                                   | 83.3%  | 69.2%  | "[Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)", Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang, arXiv:1711.09349 |
| Part-aligned（Inception V1, OpenPose) | 84.4%  | 69.3%  | "[Part-Aligned Bilinear Representations for Person Re-identification](https://arxiv.org/abs/1804.07094)", Yumin Suh, Jingdong Wang, Siyu Tang, Tao Mei, Kyoung Mu Lee, arXiv:1804.07094 |
| GP-reID                               | 85.2%  | 72.8%  | "[Re-ID done right: towards good practices for person re-identification](https://arxiv.org/abs/1801.05339)", Jon Almazan, Bojana Gajic, Naila Murray, Diane Larlus, arXiv:1801.05339 |
| SPreID (Res-152)                      | 85.95% | 73.34% | "[Human Semantic Parsing for Person Re-identification](https://arxiv.org/abs/1804.00216)", Kalayeh, Mahdi M., Emrah Basaran, Muhittin Gokmen, Mustafa E. Kamasak, and Mubarak Shah, CVPR2018 |
| MGN                                   | 88.7%  | 78.4%  | "[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438)" Wang, Guanshuo, Yufeng Yuan, Xiong Chen, Jiwei Li, and Xi Zhou. arXiv:1804.01438 |