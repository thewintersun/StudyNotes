

### Re-ranking Person Re-identification with k-reciprocal Encoding

地址：https://arxiv.org/abs/1701.08398 

作者：Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li

出版：CVPR 2017

代码：https://github.com/zhunzhong07/person-re-ranking 

知乎解释：https://www.zhihu.com/question/271308170/answer/361943914



Re-Ranking的思想基于这么一个假设：

*if a gallery image is similar to the probe in the* k*-reciprocal nearest neighbors, it is more likely to be a true match.* 

简单的解释就是，你根据probe搜索出来的candidate对象，根据这些candidate对象选择k个nearest，如果包含你的probe，那它的可能性更大一些－True match。

公式比较简单，理解为 A<->B 互为最近邻（作者称为k-倒序）：

![1563961716246](D:\Notes\raw_images\1563961716246.png)

这就是重排序（Re-Ranking），来看一张图：

![1563961165641](D:\Notes\raw_images\1563961165641.png)

可以看到，带绿框（P+）的candidate计算最近邻的结果都包含probe，重排序就往前提升，很巧，这些都是正样本，这是很直观的一个验证。

论文提到了一个扩展R*计算：

![1563961221303](D:\Notes\raw_images\1563961221303.png)

 相当于对于前面的公式的扩展，对于top-k无法query到的candidate，通过这种方法扩展，具体不展开，自己理解下公式的约束。

> **总结算法步骤：**

1）已知candidate $g_i$（上面一排）和Probe的距离为$D_i$；

2）对每个$g_i$，计算与Probe的 Jaccard 距离，计为$D_j$；

![1563961627077](D:\Notes\raw_images\1563961627077.png)

3）加权$D_i$，$D_j$，得到最终的距离，即为Re-Ranking的过程；

示意如下：

![1563961423261](D:\Notes\raw_images\1563961423261.png)

Figure 2. Proposed re-ranking framework for person re-identification. Given a probe p and a gallery, the appearance feature and k-reciprocal feature are extracted for each person. Then the original distance $d$ and Jaccard distance $d_J$ are calculated for each pair of the probe person and gallery person. The final distance $d^∗$ is computed as the combination of $d$ and $d_J$ , which is used to obtain the proposed ranking list.

其中$D_i$用的是马氏距离，$D_j$用的是Jaccard距离。

当然 Re-Ranking 的方法有很多，这里提到的 k-reciprocal Encoding 只是其中一种，至于能够降低计算量的编码方法，重点理解应该在如何降低计算量上，而不是Re-Ranking本身。 

 [rerank.py](..\codes\rerank.py) 