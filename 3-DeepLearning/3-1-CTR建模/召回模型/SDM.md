## SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

论文地址：https://arxiv.org/abs/1909.00385

作者：Fuyu Lv, Taiwei Jin, Changlong Yu, Fei Sun, Quan Lin, Keping Yang, Wilfred Ng

机构：Alibaba Group， 香港科技大学

发表：CIKM 2019

代码：https://github.com/alicogintel/SDM

数据：**JD Dataset:** [raw data](https://drive.google.com/open?id=19PemKrhA8j-RZj0i20_j4ERcnzaxl5JZ), [train and test data](https://drive.google.com/open?id=1pam-_ojsKooRLVeOXEvbh3AwJ6S4IZ7B) in the paper (tfrecord). The schema of raw data is shown in data/sample_data/.



### 摘要

Capturing users' precise preferences is a fundamental problem in large-scale recommender system. Currently, item-based Collaborative Filtering (CF) methods are common matching approaches in industry. However, they are not effective to model dynamic and evolving preferences of users. In this paper, we propose a new sequential deep matching (SDM) model to ==capture users' dynamic preferences by combining short-term sessions and long-term behaviors==. Compared with existing sequence-aware recommendation methods, we tackle the following two inherent problems in real-world applications: 

(1) there could exist multiple interest tendencies in one session.  

(2) long-term preferences may not be effectively fused with current session interests. 

Long-term behaviors are various and complex, hence those highly related to the short-term session should be kept for fusion. We propose to encode behavior sequences with two corresponding components: multi-head self-attention module to capture multiple types of interests and long-short term gated fusion module to incorporate long-term preferences. Successive items are recommended after matching between sequential user behavior vector and item embedding vectors. 

Offline experiments on real-world datasets show the superior performance of the proposed SDM. Moreover, SDM has been successfully deployed on online large-scale recommender system at Taobao and achieves improvements in terms of a range of commercial metrics.



### 介绍

<img src="D:\Notes\raw_images\image-20210909120921793.png" alt="image-20210909120921793" style="zoom:80%;" />

<img src="D:\Notes\raw_images\image-20210909121010514.png" alt="image-20210909121010514" style="zoom:80%;" />

<img src="D:\Notes\raw_images\image-20210909121043806.png" alt="image-20210909121043806" style="zoom:80%;" />

<img src="D:\Notes\raw_images\image-20210909121104767.png" alt="image-20210909121104767" style="zoom:80%;" />

<img src="D:\Notes\raw_images\image-20210909121153789.png" alt="image-20210909121153789" style="zoom:80%;" />