### A Simple Pooling-Based Design for Real-Time Salient Object Detection
- 论文地址：https://arxiv.org/abs/1904.09569
- 作者： Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Jiashi Feng, Jianmin Jiang
- 机构： 南开大学
- Accepted by CVPR2019
- PaperWeekly介绍： https://mp.weixin.qq.com/s/urgkUcu2ZWQMGPZdArWzYg 
- 代码地址： https://github.com/backseason/PoolNet 


#### 介绍
这是一篇发表于 CVPR 2019 的关于显著性目标检测的 paper，在 U 型结构的特征网络中，高层富含语义特征捕获的位置信息在自底向上的传播过程中可能会逐渐被稀释，另外卷积神经网络的感受野大小与深度是不成正比的。
目前很多流行方法都是引入 Attention（注意力机制），但是本文是基于 U 型结构的特征网络研究池化对显著性检测的改进，具体步骤是引入了两个模块GGM (Global Guidance Module，全局引导模块) 和 FAM (Feature Aggregation Module，特征整合模块)，进而锐化显著物体细节，并且检测速度能够达到 30FPS。因为这两个模块都是基于池化做的改进所以作者称其为 PoolNet，并且放出了源码：https://github.com/backseason/PoolNet

<img src='https://mmbiz.qpic.cn/mmbiz_png/VBcD02jFhgmHBJZu2BrUlanYYxJ6koQYrw1TsVKkgLBI0EoPicV4B0hzK5SbXuY4f39AYr0eO6uKu2al6Rs4YuA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1'/>
