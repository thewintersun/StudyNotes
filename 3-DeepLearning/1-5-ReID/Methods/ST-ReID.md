### Spatial-Temporal Person Re-identification

论文地址：https://arxiv.org/pdf/1812.03282v1.pdf

作者：Guangcong Wang , Jianhuang Lai, Peigen Huang, Xiaohua Xie

机构：School of Data and Computer Science, Sun Yat-sen University, China
Guangdong Key Laboratory of Information Security Technology
Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education

发表：AAAI 2019

代码地址：https://github.com/Wanggcong/Spatial-Temporal-Re-identification

知乎文章： https://zhuanlan.zhihu.com/p/52274204



#### 摘要

目前大多数的人重新识别（ReID）方法都忽略了时空约束（neglect a spatial-temporal constraint.）。因此提出了 ==a novel two-stream spatial-temporal person ReID (st-ReID) framework==，同时挖掘视觉语义信息和时空信息。为此还引入了 ==Logistic Smoothing 的联合相似度度量方法==来整合这两种异构信息。为了拟合复杂的时空概率分布，我们开发了一种==快速的Histogram-Parzen（HP）方法==。结果借助于时空约束，st-ReID模型消除了许多不相关的图像，从而缩小了图片库。 

our st-ReID method achieves rank-1 accuracy of 98.1% on Market-1501 and 94.4% on DukeMTMC-reID, improving from the baselines 91.2% and 83.8%, respectively, outperforming all previous state-of-the-art methods by a large margin.



#### 方法

三部分组成，visual feature stream, a spatial-temporal stream, and a joint metric sub-module。

![1566289755115](D:\Notes\raw_images\1566289755115.png)

Figure 3: The proposed two-stream architecture. It consists of three sub-modules, i.e., a visual feature stream, a spatial-temporal stream, and a joint metric sub-module. (Best viewed in color)

##### **Visual Feature Stream**

ResNet50+PCB，很简单有效的模型，可以在Market上把rank1直接怼到91。

##### Spatial-Temporal Stream

 我们估计spatial-temporal分布通过使用非参数估计方法，即Parzen Window方法。 但是，直接估计PDF会花费很多时间，因为有太多了spatial-temporal数据点。
为了减轻昂贵的计算问题，我们开发Histogram-Parzen方法。 那就是我们先估计spatial-temporal直方图，然后使用 Parzen Window方法使其平滑。

Spatial-Temporal Histogram

Let $(ID_i, c_i, t_i)$ and $(ID_j, c_j, t_j)$  $(t_i < t_j )$ denote the identity labels, camera IDs, timestamps of two images $I_i$ and $I_j$ , respectively.
$$
\hat{p}(y=1|k,c_i,c_j) = \frac{n_{c_i,c_j}^k}{\sum_l n_{c_i,c_j}^l}
$$
用某个时间区间内的历史样本数量（分子）除以总样本数（分母）得到一个时空概率，这里把统计出来的时空概率按迁移时间$t$分成了很多个bin（100帧一个bin），每个bin里的样本时空概率是相同的，求解时空概率时，只需要看 $t$ 是落在哪个bin内，比如说是第 $k$ 个bin，就返回这个bin的极大似然估计结果。这种做法的测试时效率会比较高一点。

代码里在统计历史样本数量的时候，==还对一个人在某个摄像头的多次出现的时间做了平均，这样可以避免某个人在摄像头中出现时间太长引入的统计误差==。

Parzen Window Smooth
$$
p(y=1|k,c_i,c_j) = \frac{1}{Z}\sum_t \hat{p}(y=1|k,c_i,c_j) K(1-k)
$$
$$
K(x) = {1 \over \sigma\sqrt{2\pi} }\,e^{- {{(x)^2 \over 2\sigma^2}}}
$$

看着很复杂，但是记住一个目标就是smooth，当你要计算第 $k$ 个bin的分数时，还会统计周围几个bin的分数（看了代码，统计的是全部 bin 的分数），用 $K(·)$ 这个函数求第 $l$ 个bin的平滑权重，==这个$K(·)$是一个高斯函数，也就是离得越远权重越小==，实际上这种平滑TFusion代码里也有做，只是我们觉得太细枝末节了没有讲。					

**Joint Metric**

视觉分数和时空分数要怎么结合呢？我们在TFusion中也讲了，直接相乘是不行的，在这篇论文里也分析了，两个分数，一高一低相乘会低与两个中等的分数，这不太符合需求。所以这篇论文做了一个操作==将分数限制在一个范围，并对最终分数做sigmoid激活，这样相乘效果就比较好了==。

Laplace smoothing

Laplace smoothing is a technique which is widely used to estimate a prior probability in Naive Bayes
$$
p_{\lambda(Y=d_k)}=\frac{m_k+\lambda}{M+D\lambda}
$$
$\frac{m_k}{M}$ 是类别为 $d_k$ 的样本在所有样本中的比例作为一个先验。对于这个未平滑的时空，分子加一个$\lambda$，分母加一个$D\lambda$，这样  $p_{\lambda(Y=d_k)}$ 就被限制在 $[\frac{1}{D}, \frac{m_k}{M}]$ 这个区间里，这样就可以防止出现比较小的时空分数。

_话说我在代码里没找到这部分逻辑，如果有找到的同学欢迎留言。从代码中统计历史数据的逻辑看，也可能是指第 k个 person id，为 $d_k$ 的样本在所有样本中的比例作为一个先验。但测试集没有 $d_k$ 的 person id啊，搞不懂…_

**Logistic function**

正常的带超参数的sigmoid激活：
$$
f(x;\lambda;\gamma)= \frac{1}{1+\lambda e^{-\gamma x}}
$$
于是最终的分数
$$
p_{joint} = f(s;\lambda_0,\gamma_0)f(p_{st};\lambda_1,\gamma_1)
$$
==激活是对视觉和时空都做的，并且两个分数各有两个超参数来做权衡==（这样就能通过调参把分数调上去了）。

For notation simplicity, we use $p_{joint}$,  $s$  and  $p_{st}$ to denote $p (y = 1|x_i,x_j, k, c_i, c_j)$ , $ s(x_i, x_j) $ and $p(y = 1|k, c_i, c_j)$ respectively. 

According to Eqn. (1) and (3), we can see that $s \in (-1, 1)$  is shrunk by the logistic function like the Laplace smoothing, but not so much. Differently,  $p_{st} \in (0, 1)$ is truncated and lifted up largely. Even the spatial-temporal probability $p_{st}$ is close to zero, $f(p_{st};\lambda_1, \gamma_1) \ge f(0) = \frac{1}{1+\lambda}$.  With the logistic smoothing, Eqn. (9) is robustt o rare events. 

时空概率相对于视觉相似度是不稳定的。Besides, using the logistic function to transform the similarity score (spatial temporal probability) into a binary classification probability (positive pair or negative pair) is intuitive and self-evident as described in Observation 2.

#### 实验

实验数据集就是market和duke，没有常见的cuhk03，原因在TFusion中也说过，只有这两个数据集和grid数据集有时间数据。实验效果还是挺好的，多模态方法好好调参完虐纯视觉方法。实验中也和SOTA方法做了对比，包括我们的TFusion，当时做了一个source和target都一样的实验，但是还是用无监督的时空估计方法，所以准确率还是AAAI这篇比较高。

![1566373919447](D:\Notes\raw_images\1566373919447.png)

然而他们说:  Therefore, TFusion-sup actually does not investigate how to estimate the spatial-temporal probability distribution and how to model the joint probability of the visual similarity and the spatial-temporal probability distribution.

这一段我们是不认的，仔细读我们的论文就知道我们是有做这两个事情的，并且是在无监督条件下。当然这篇能在有监督下把多模态融合和时空模型估计做到极致也是很有价值的。