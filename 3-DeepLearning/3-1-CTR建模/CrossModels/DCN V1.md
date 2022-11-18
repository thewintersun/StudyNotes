## Deep & Cross Network for Ad Click Predictions

地址：https://arxiv.org/abs/1708.05123

作者：Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang

机构：Stanford University， Google Inc.

发表： In Proceedings of AdKDD and TargetAd, Halifax, NS, Canada, August, 14, 2017, 7 pages



### 摘要

Feature engineering has been the key to the success of many prediction models. However, the process is non-trivial and often requires manual feature engineering or exhaustive searching. DNNs are able to automatically learn feature interactions; however, they generate all the interactions implicitly, and are not necessarily efficient in learning all types of cross features. 

In this paper, we propose the Deep & Cross Network (DCN) which keeps the benefits of a DNN model, and beyond that, it ==introduces a novel cross network that is more efficient in learning certain bounded-degree feature interactions==. In particular, DCN explicitly ==applies feature crossing at each layer==, requires no manual feature engineering, and adds negligible extra complexity to the DNN model. Our experimental results have demonstrated its superiority over the state-of-art algorithms on the CTR prediction dataset and dense classification dataset, in terms of both model accuracy and memory usage.



### 介绍

模型结构如下，共分为4个部分，分别为 Embedding and Stacking Layer（特征预处理输入）、Cross network（自动化特征显式交叉）、Deep network（特征隐式交叉）和Combination output Layer（输出）。

![image-20211029151955105](D:\Notes\raw_images\image-20211029151955105.png)

<img src="D:\Notes\raw_images\image-20211029152319708.png" alt="image-20211029152319708" style="zoom:67%;" />

#### 1.1 Embedding and Stacking Layer

常规操作，首先针对原始特征进行预处理，其中类别特征（Sparse feature）可以通过二值化处理，然后进行特征Embedding，将高维稀疏特征转化为低维稠密的实值向量（Embedding vec），再拼接其他连续特征（Dense feature）作为模型的输入。
$$
\begin{aligned} X_0=[X_{embed,1}^T,\dots,X_{embed,,}^T,X_{dense}^T] \end{aligned} \qquad (1) \\
$$
其中 $X_{embed,i}^T $代表特征Embedding vec，$X_{dense}^T $代表连续实值特征向量。为了方便与下面的介绍保持统一，假设 $X_0 \in \mathbb{R}^d ，X_0=[x_1,x_2,\dots,x_d] $。

#### 1.2 Cross Network

这个部分是模型的核心，数学表达式如下：
$$
\begin{aligned} X_{l+1}=X_0X_{l}^TW_{l}+b_{l}+X_{l}=f(X_{l},W_{l},b_{l})+X_{l} \end{aligned} \qquad (2) \\
$$
其中 $X_{l},X_{l+1} \in \mathbb{R}^d $分别代表Cross Network的第 $l,l+1 $ 层的输出，$W_{l},b_{l} \in \mathbb{R}^d $分别为该层的参数与偏置项。因为 $f(X_{l},W_{l},b_{l})=X_{l+1}-X_{l} $ ，所以函数 $f:\mathbb{R}^d \mapsto \mathbb{R}^d $是拟合$ X_{l+1} $与 $X_{l} $ 的残差，这个思想与Deep Crossing一致。

<img src="D:\Notes\raw_images\image-20211029153330289.png" alt="image-20211029153330289" style="zoom:80%;" />

公式（2）对应的图形化表示如Fig 2，图中所有的向量维度都是一样的。利用 $X_0 $与 $X^{\prime} $做向量外积得到所有的元素交叉组合，层层叠加之后便可得到任意有界阶组合特征，当cross layer叠加 l 层时，交叉最高阶可以达到 l+1 阶，参考[5]可以举例说明：

为了方便起见，首先将 b 设置为零向量，令$ X_0=\left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] $，那么
$$
\begin{aligned} X_1={} & X_0X_0^{\prime}W_0+X_0 \\ ={} &  \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \left[x_{0,1}x_{0,2}\right] \left[\begin{matrix}w_{0,1}\\w_{0,2}\end{matrix}\right] +  \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}x_{0,1}^2,x_{0,1}x_{0,2}\\x_{0,2}x_{0,1},x_{0,2}^2\end{matrix}\right] \left[\begin{matrix}w_{0,1}\\w_{0,2}\end{matrix}\right] + \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}\\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2\end{matrix}\right] + \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}w_{0,1}{\color{Red}{x_{0,1}^2}}+w_{0,2}{\color{Red}{x_{0,1}x_{0,2}}}+{\color{Red}{x_{0,1}}}\\ w_{0,1}{\color{Red}{x_{0,2}x_{0,1}}}+w_{0,2}{\color{Red}{x_{0,2}^2}}+{\color{Red}{x_{0,2}}}\end{matrix}\right] \end{aligned} \qquad (3) \\
$$
继续计算 X_2，有：
$$
\begin{aligned} X_2={} & X_0X_1^{\prime}W_1+X_1 \\ ={} & \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \left[w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1}, \quad  w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\right] \left[\begin{matrix}w_{1,1}\\w_{1,2}\end{matrix}\right] \\  +{} &  \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1} \\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix} w_{0,1}x_{0,1}^3+w_{0,2}x_{0,1}^2x_{0,2}+x_{0,1}^2, \quad  w_{0,1}x_{0,2}x_{0,1}^2+w_{0,2}x_{0,2}^2x_{0,1}+x_{0,2}x_{0,1} \\ w_{0,1}x_{0,1}^2x_{0,2}+w_{0,2}x_{0,1}x_{0,2}^2+x_{0,1}x_{0,2}, \quad  w_{0,1}x_{0,2}^2x_{0,1}+w_{0,2}x_{0,2}^3+x_{0,2}^2 \end{matrix}\right] \left[\begin{matrix}w_{1,1}\\w_{1,2}\end{matrix}\right] \\ +{} &  \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1} \\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix} w_{0,1}w_{1,1}{\color{Red}{x_{0,1}^3}}+w_{0,2}w_{1,1}{\color{Red}{x_{0,1}^2x_{0,2}}}+w_{1,1}{\color{Red}{x_{0,1}^2}} +  w_{0,1}w_{1,2}{\color{Red}{x_{0,2}x_{0,1}^2}}+w_{0,2}w_{1,2}{\color{Red}{x_{0,2}^2x_{0,1}}}+w_{1,2}{\color{Red}{x_{0,2}x_{0,1}}} \\ w_{0,1}w_{1,1}{\color{Red}{x_{0,1}^2x_{0,2}}}+w_{0,2}w_{1,1}{\color{Red}{x_{0,1}x_{0,2}^2}}+w_{1,1}{\color{Red}{x_{0,1}x_{0,2}}} +  w_{0,1}w_{1,2}{\color{Red}{x_{0,2}^2x_{0,1}}}+w_{0,2}w_{1,2}{\color{Red}{x_{0,2}^3}}+w_{1,2}{\color{Red}{x_{0,2}^2}} \end{matrix}\right] \\ +{} &  \left[\begin{matrix}w_{0,1}{\color{Red}{x_{0,1}^2}}+w_{0,2}{\color{Red}{x_{0,1}x_{0,2}}}+{\color{Red}{x_{0,1}}} \\ w_{0,1}{\color{Red}{x_{0,2}x_{0,1}}}+w_{0,2}{\color{Red}{x_{0,2}^2}}+{\color{Red}{x_{0,2}}}\end{matrix}\right] \\ \end{aligned} (4) \\
$$
从公式（3）（4）的标红处可以看出，==当cross layer叠加 l 层时，交叉最高阶可以达到 l+1 阶，并且包含了所有的交叉组合，这是DCN的精妙之处==。

复杂性分析：
一般来说，要对特征进行高阶显式交叉，一定会加大模型的参数量。在DCN中，假设Cross Layer有$ L_c$  层，最底层输入 $X_0 $为 $d$ 维。因为每一层仅有 $W,b$ 参数，所以模型参数量会额外增加 $ d \times L_c \times 2 $ 个。

因为$ X_0X_{l}^{\prime}W_{l}=X_0(X_{l}^{\prime}W_{l}) $，先计算 $X_{l}^{\prime}W_{l} $ 得到一个标量，然后再与$ X_0$ 相乘，在时间与空间上计算效率都得到提升，最终$ L_c $层 Cross Layer 的时空复杂度均为 $O(dL_c) $，也就是说==时空复杂度随着输入与层数线性增长==，这是非常好的性质。

让我们更进一步分析一下Cross Layer的设计理念，通常来说当两个向量的外积之后，往往想到的是再利用一个矩阵来对结果进行压缩变换，假设向量外积之后为$ d \times n $维矩阵，那么为了将结果变换为 $d $ 维向量，需要使用的参数矩阵为 $ n \times d $ 维，不仅参数量变多了，而且矩阵相乘的运算复杂度高达三次方。

This efficiency benefits fromthe rank-one property of $ X_0X_1^{\prime}$, which enables us to generate all cross termswithout computing or storing the entire matrix.

参数量过多很容易造成过拟合，参数量的适当精简反而可以提高模型的泛化能力与鲁棒性。如Cross Layer中使用向量而不是矩阵来对结果进行变换，本质上是通过==参数共享的模式减少参数量==。==共享参数能够对样本外数据仍有较好的泛化能力，且能够对噪声数据带来的参数变化进行纠正==。

#### 1.3 Deep Network

与之前的Wide&Deep一样，这个部分是简单的DNN结构。可以表示为：
$$
\begin{aligned} h_{l+1}=f(W_{l}h_{l}+b_{l}) \end{aligned} \qquad (5) \\
$$
其中 $h_{l},h_{l+1} $ 分别为第 $ l,l+1$ 层的输出，W_l 与 b_l 为参数与偏置项， f 为 relu激活函数。假设DNN中的隐层节点为 m 个，共有 L_d 层，且最底层输入为 d 维，那么DNN部分参数量为 $ d \times m + m + (m^2+m) \times (L_d-1)$ 。

#### 1.4 Combination Output Layer

将Cross Network与Deep Network部分的输出进行简单拼接，通过激活函数作为最后的输出。
$$
\begin{aligned} p=\sigma\left({[X_{L_1}^T,h_{L_2}^T]}W_{logits}\right) \end{aligned} \qquad (6) \\
$$
其中$ X_{L_1}^T \in \mathbb{R}^d ，h_{L_2}^T \in \mathbb{R}^m $分别为Cross Network与Deep Network的输出，$W_{logits} \in \mathbb{R}^{d+m} $是变换矩阵，$\sigma $为 sigmoid 激活函数。
模型使用的Loss函数为 logloss，并且加入了正则项：
$$
\begin{aligned} loss = -\frac{1}{N}\sum_{i=1}^Ny_ilog(p_i)+(1-y_i)log(1-p_i)+ \lambda \sum ||W||^2 \end{aligned} \qquad (7) \\
$$

### 实验结果

作者使用 Criteo 数据集进行实验对比，对比模型有 LR、FM、DNN、Deep Crossing。至于为什么没有Wide&Deep模型进行对比，在原文中提到，因为Wide部分需要手工构造合适的交叉特征，这需要领域知识来对特征进行选择，不方便作为对比试验。
W&D. Different than DCN, its wide component takes as input raw sparse features, and relies on exhaustive searching and domain knowledge to select predictive cross features. We skipped the com- parison as no good method is known to select cross features.

几个模型的对比结果如下，结果来看DCN的确表现更优。

<img src="D:\Notes\raw_images\image-20211029172010340.png" alt="image-20211029172010340" style="zoom:80%;" />


最后，作者设计实验观察不同Cross Layer层数与隐层节点数对结果带来的影响。

<img src="D:\Notes\raw_images\image-20211029172341713.png" alt="image-20211029172341713" style="zoom:80%;" />

![image-20211029172410989](D:\Notes\raw_images\image-20211029172410989.png)