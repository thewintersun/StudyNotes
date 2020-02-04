### CUHK Person Re-identification Datasets

官网地址: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html

| Name       | Description                                                  | Links                                                        |
| :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **CUHK01** | 971 identities, 3884 images, manually cropped                | [DOWNLOAD](https://docs.google.com/spreadsheet/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA)   [PAPER](http://www.ee.cuhk.edu.hk/~rzhao/project/transfer_accv12/liZWaccv12.pdf) |
| **CUHK02** | 1816 identities, 7264 images, manually cropped               | [DOWNLOAD](https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHZtSGIwTnVDUEdWMFktQWU2bTZ0N3c6MA#gid=0)   [PAPER](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Locally_Aligned_Feature_2013_CVPR_paper.pdf) |
| **CUHK03** | 1360 identities, 13164 images, manually cropped + automatically detected | [DOWNLOAD](https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0)   [PAPER](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) |

**简要描述：** 

MATLAB数据文件格式，==1467个行人==，收集自==香港中文大学==The Chinese University of Hong Kong校园内的==10个(5对)==不同的摄像头。数据集结构由三部分组成：

- "detected"：行人框由pedestrian detector绘出，5x1 cell，分别由5对摄像头组收集得到。

```html
  --   843x10 cell，收集自摄像头组pair 1，行数为行人索引，前5列和后5列分别来自同一组的不同摄像头。cell内每个元素为一幅 MxNx3 的行人框图像(uint8 数据类型)，个别图像可能空缺，为空集。

  --   440x10 cell，收集自摄像头组pair 2，其它同上。

  --   77x10 cell，收集自摄像头组pair 3，其它同上。

  --   58x10 cell，收集自摄像头组pair 4，其它同上。

  --   49x10 cell，收集自摄像头组pair 5，其它同上。
```

-  "labeled" ：5x1 cell，行人框由人类标注，格式和内容大致和上面的"detected"相同。

-  "testsets" ：20x1 cell，测试协议。由20个 100x2 double类型矩阵组成。

      --   100x2 double，100行代表100个测试样本，第1列为摄像头pair索引，第2列为行人索引。



**测试协议：**

CUHK-03的测试协议有两种。

- 第一种为旧的版本，参见数据集中的'testsets'测试协议。具体地说，即随机选出100个行人作为测试集，1160个行人作为训练集，100个行人作为验证集（这里总共1360个行人而不是1467个，这是因为实验中没有用到摄像头组pair 4和5的数据），重复二十次。这种测试协议是single-shot setting.

- 第二种测试协议, 类似于Market-1501，它将数据集分为包含767个行人的训练集和包含700个行人的测试集。在测试阶段，我们随机选择一张图像作为query，剩下的作为gallery，这样的话，对于每个行人，有多个ground truth在gallery中。（新测试协议可以参考https://github.com/zhunzhong07/person-re-ranking） 

  

**数据集下载地址：**

Google Drive: 

https://drive.google.com/file/d/0BxJeH3p7Ln48djNVVVJtUXh6bXc/edit?usp=sharing

Baidu Cloud Disk ( password: rhjq ):

http://pan.baidu.com/s/1mgklxSc



### State-of-The-Art

![1564023167055](C:\Users\j00496872\Desktop\Notes\raw_images\1564023167055.png)

https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP

|                   | Labeled    | Labeled | detected   | detected  |                                                              |
| ----------------- | ---------- | ------- | ---------- | --------- | ------------------------------------------------------------ |
| **Methods**       | **Rank@1** | **mAP** | **Rank@1** | **mAP**   | **Reference**                                                |
| DaRe              | 58.1%      | 53.7%   | 55.1%      | 51.3%     | "[Resource Aware Person Re-identification across Multiple Resolutions](http://www.cs.cornell.edu/~gaohuang/papers/Anytime-ReID.pdf)", Yan Wang, Lequn Wang, Yurong You, Xu Zou, Vincent Chen. CVPR 2018 |
| TriNet+RE         | 58.14%     | 53.83%  | 55.50%     | 50.74%    | "[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)", Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang, arXiv 2017 |
| TriNet+RR+RE      | 63.93%     | 65.05%  | 64.43%     | 64.75%    | "[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)", Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang, arXiv 2017 |
| PCB (RPP)         | -          | -       | 63.7%      | 57.5%     | "[Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)](https://arxiv.org/pdf/1711.09349.pdf)", Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang, arXiv 2017 |
| HPM+HRE           | -          | -       | 63.2%      | 59.7%     | "[Horizontal Pyramid Matching for Person Re-identification](https://arxiv.org/pdf/1804.05275.pdf)", Yang Fu, Yunchao Wei, Yuqian Zhou, Honghui Shi, arXiv 2018 |
| ==DGNet==         | -          | -       | 65.6%      | 61.1%     | "[Joint Discriminative and Generative Learning for Person Re-identification](https://arxiv.org/abs/1904.07223)", Zhedong Zheng, Xiaodong Yang, Zhiding Yu, Liang Zheng, Yi Yang and Jan Kautz, CVPR 2019 (Oral) |
| MGN               | 68.0%      | 67.4%   | 66.8%      | 66%       | "[Learning Discriminative Features with Multiple Granularity for Person Re-Identification](https://arxiv.org/pdf/1804.01438.pdf)", Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou, arXiv 2018 |
| ==DaRe(R)+RE+RR== | 72.9%      | 73.7%   | 69.8%      | ==71.2%== | "[Resource Aware Person Re-identification across Multiple Resolutions](http://www.cs.cornell.edu/~gaohuang/papers/Anytime-ReID.pdf)", Yan Wang, Lequn Wang, Yurong You, Xu Zou, Vincent Chen. CVPR 2018 |

- RR (Re-ranking) Re-ranking person re-identification with k-reciprocal encoding. Z. Zhong, L. Zheng, D. Cao, and S. Li. CVPR 2017. [[code\]](https://github.com/zhunzhong07/person-re-ranking)
- RE (Random Erasing) Random erasing data augmentation. Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang. arXiv, 2017. [[Code\]](https://github.com/zhunzhong07/Random-Erasing)