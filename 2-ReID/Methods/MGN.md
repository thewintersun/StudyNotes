#### Learning Discriminative Features with Multiple Granularities for Person Re-Identification

论文地址：https://arxiv.org/pdf/1804.01438.pdf

作者：Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou

机构：上海交通大学，云从科技

代码地址：https://github.com/seathiefwang/MGN-pytorch



**背景导读**

人脸识别技术经过进几年的发展，已较为成熟，在众多的场景与产品中都已有应用，但人脸识别技术只能用到人体的人脸信息，放弃了人体其他重要信息的利用，例如==衣着、姿态、行为==等，另外在应用时必须要有清晰的人脸正面照片，但在很多场景下无法满足要求，例如==低头、背影、模糊身形、帽子遮挡==等等。而跨镜追踪（ReID）技术正好能够弥补人脸识别的这些不足，行人重识别能够根据行人的穿着、体态、发型等信息认知行人。这将人工智能的认知水平提高到一个新的阶段，现在跨镜追踪已成为人工智能领域的重要研究方向。但现有的研究成果还不是很成熟，离实际商用的要求还有一定距离。



**定义：**

==跨镜追踪技术（Person Re-Identification，简称 ReID）==是现在计算机视觉研究的热门方向，是利用计算机视觉技术判断图像或者视频序列中是否存在特定行人的技术。该技术可以作为人脸识别技术的重要补充，可以对无法获取清晰拍摄人脸的行人进行跨摄像头连续跟踪，增强数据的时空连续性。该技术可以广泛应用于视频监控、智能安保、智能商业等领域。



**难点：**

ReID技术与人脸识别技术类似，存在较多的困难点需要克服，例如==光线、遮挡、图片模糊==等客观因素。另外，==行人的穿着多样，同一人穿不同的衣服==，==不同的人穿相似的衣服==等等也对ReID技术提出更高的要求。行人的姿态多变导致人脸上广泛使用的对齐技术也在ReID失效。行人的数据获取难度远远大于人脸识别数据获取难度，而行人的信息复杂程度又远远大于人脸，这两个因素叠加在一起使得ReID的算法研究变得更加困难，也更加重要。通过算法的有效设计，降低对数据依赖来实现ReID效果的突破是现在业内的共识。

提出通过==融合行人的全局信息==以及==具有辨识力的多粒度局部信息==的思路，为解决ReID问题提供了一个非常不错的思路。本次提出的方案有几大优势: 

- 结构精巧：该方案实现了端到端的直接学习，并没有增加额外的训练流程，

- 多粒度：融合了行人的整体信息与==有区分度的多粒度细节信息==，

- 关注细节：模型真正懂得什么是人，模型会把注意力放在膝盖，衣服商标等能够显著区分行人的一些核心信息上。

  

**研究成果:**

Market-1501，CUHK03，DukeMTMC-reID是衡量ReID技术的最主流的数据集。首位命中率（Rank-1 Accuracy）、平均精度均值（Mean Average Precision，mAP）是衡量ReID技术水平的核心指标。云从科技在这三个数据集中都刷新了业内最好的水平，在Market-1501数据集的Rank-1 Accuracy达到惊人的95.7%，使用Re-Ranking 技术后更是达到96.6%。==mAP是更加全面衡量ReID算法效果的指标，它要求将检索图片与被检索图片都匹配正确，而不止首位命中。==云从科技此次将Market-1501的mAP指标将现在最好的水平提高了近5%，达到86.9%，使用Re-Ranking技术之后更是达到了94.2 %。



**文章导读**

作者表示结合全局与局部特征是提高提取行人关键可分辨信息的重要方法。之前的局部特征提取的方法专注在基于位置的预先定义的语义信息进行提取，导致训练难度提高，同时在复杂场景的鲁棒性并不尽如人意。而==作者新设计了一个多分支的端到端的深度网络，使得不同级别的网络分支能够关注不同粒度的分辨信息，也能够有效兼顾整体信息==。损失函数部分，作者表示为了充分体现网络的真实潜力，该文章中只使用了在深度学习中非常常见的Softmax Loss 与 Triplet Loss。

多粒度网络（Multiple Granularity Network, MGN），包含三个分支，一个分支用来学习全局特征表达，两个分支用来学习局部特征表达。

本文不是在预定义的语义区域上进行特征学习，而是将图像均匀地分成几个条纹，并在不同的局部分支中的部件数量不同，具有多个粒度的表示。如下图所示，一个全局信息，2个局部信息，每个局部信息划分的块数不同（2，3）。

![1564730943452](D:\Notes\raw_images\1564730943452.png)

作者的对多粒度的解析如图所示，从左到右是人体部分从粗粒度到精细粒度的过程。左边三张是完整的行人图片，中间是将行人图片分割为上下两部分，最有右边是将行人图片分成上中下三个部分。

Part-based methods for person Re-ID can be divided into three main pathways according to their part locating methods: 

1. Locating part regions with strong structural information such as empirical knowledge about human bodies [8, 21, 36, 43] or strong learning-based pose information [33, 44]; 
2. Locating part regions by region proposal methods [19, 41]; 
3. Enhancing features by middle-level attention on salient partitions [22, 24, 25, 45]. 

However, obvious limitations impede the effectiveness of these methods. 

1. First, pose or occlusion variations can affect the reliability of local representation. 
2. Second, these methods almost only focus on specific parts with fixed semantics, but cannot cover all the discriminative information. 
3. Last but not least, most of these methods are not end-to-end learning process, which increases the complexity and difficulty of feature learning.
   

**网路结构图**

多粒度网络(Multiple Granularity Network,MGN) 如上图所示，该结构的基础网络部分采用业内最为常用的Resnet50。根据对Resnet50网络以及跨镜追踪的深刻分析，作者创新性地对Resnet50进行了合理的修改，使用Resnet50前三层提取图像的基础特征，而在高层次的语意级特征作者设计了3个独立分支。如图所示，第一个分支负责整张图片的全局信息提取，第二个分支会将图片分为上下两个部分提取中粒度的语意信息，第三个分支会将图片分为上中下三个部分提取更细粒度的信息。这三个分支既有合作又有分工，前三个低层权重共享，后面的高级层权重独立，这样就能够像人类认知事物的原理一样即可以看到行人的整体信息与又可以兼顾到多粒度的局部信息。

![1564731591147](D:\Notes\raw_images\1564731591147.png)

同时文章对损失函数部分也进行了精心而巧妙的设计。==三个分支最后一层特征都会进行一次全局MaxPooling操作==，而第二分支与第三分支还会 ==分别再进行局部的MaxPooling==，然后再将特征由2048维降为256维。最后256维特征同时用于Softmax Loss与Triplet Loss计算。另外，作者在2048维的地方添加一个额外的全局Softmax Loss，该任务将帮助网络更全面学习图片全局特征。

全局分支：res_conv4_2，接stride=2 的res_conv5_1 block下采样, 接Global Max Pooling, 接1x1卷积（带BN和ReLU）从2018降维到256，分别计算2048维（完整特征信息）的software loss（分类概念）和全部被降纬（完整信息被降维）的triplet loss。

局部分支：没有采用下采样技术，而把输出的特征映射均匀的分离到各个条纹中，其余过程与全局分支过程一样,在计算loss方面，分别计算2048维（完整特征信息）和256维（简化的特征信息）的software loss（分类概念）以及全部被降纬（完整信息被降维）的triplet loss，Part-3与Part-2差不多。

而在测试的时候==只需使用使用256维特征作为该行人的特征进行比较，无需使用2048维的特征==，使用欧氏距离作为两个行人相似度的度量。

**Loss函数**

Softmax Loss:



![1564735803046](D:\Notes\raw_images\1564735803046.png)

All the global features after reduction ${f_g^G, f_g^{P2} , f_g^{P3} }$ are trained with triplet loss to enhance ranking performances. We use the ==batchhard triplet loss== [14], an improved version based on the originals emi-hard triplet loss:

![1564735957634](D:\Notes\raw_images\1564735957634.png)

**实验结果**

![1564734745731](D:\Notes\raw_images\1564734745731.png)

![1564736290755](D:\Notes\raw_images\1564736290755.png)

![1564736303893](D:\Notes\raw_images\1564736303893.png)

