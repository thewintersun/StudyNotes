## KL散度|JS散度|Wasserstein距离

### 1. KL散度

> **KL散度**又称为**相对熵**，信息散度，信息增益。KL散度是是两个概率分布P和Q **差别**的非对称性的度量。 KL散度是用来 度量使用基于Q的编码来编码来自P的样本平均所需的额外的位元数。 典型情况下，P表示数据的真实分布，Q表示数据的理论分布，模型分布，或P的近似分布。

定义如下：

![img](http://upload-images.jianshu.io/upload_images/3596589-dc6bdef12f91d925.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为对数函数是凸函数，所以KL散度的值为非负数。

> 有时会将KL散度称为KL距离，但它并不满足距离的性质：
>
> 1. KL散度不是对称的；
> 2. KL散度不满足三角不等式。

### 2. JS散度(Jensen-Shannon)

> **JS散度**度量了两个概率分布的相似度，基于KL散度的变体，解决了KL散度非对称的问题。一般地，JS散度是对称的，其取值是0到1之间。定义如下：

![img](http://upload-images.jianshu.io/upload_images/3596589-26065ae8c3b8b87f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> KL散度和JS散度度量的时候有一个问题：
>
> 如果两个分配P,Q离得很远，完全没有重叠的时候，那么KL散度值是没有意义的，而JS散度值是一个常数。这在学习算法中是比较致命的，这就意味这这一点的梯度为0。梯度消失了。

### 3. Wasserstein距离

> **Wasserstein距离**度量两个概率分布之间的距离，定义如下：

![img](http://upload-images.jianshu.io/upload_images/3596589-72a9092dd247615f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Π(P1,P2)是P1和P2分布组合起来的所有可能的联合分布的集合。对于每一个可能的联合分布γ，可以从中采样(x,y)∼γ得到一个样本x和y，并计算出这对样本的距离||x−y||，所以可以计算该联合分布γ下，样本对距离的期望值E(x,y)∼γ[||x−y||]。在所有可能的联合分布中能够对这个期望值取到的下界$inf_{γ∼Π(P1,P2)}E(x,y)∼γ[||x−y||]$ 就是Wasserstein距离。

直观上可以把E(x,y)∼γ[||x−y||]理解为在γ这个路径规划下把土堆P1挪到土堆P2所需要的消耗。而Wasserstein距离就是在最优路径规划下的最小消耗。所以Wesserstein距离又叫Earth-Mover距离。

Wessertein距离相比KL散度和JS散度的**优势**在于：即使两个分布的支撑集没有重叠或者重叠非常少，仍然能反映两个分布的远近。而JS散度在此情况下是常量，KL散度可能无意义。

> References:
>
> 1. 维基百科[KL散度](https://zh.wikipedia.org/wiki/相对熵)
> 2. 维基百科[JS散度](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
> 3. 维基百科[Wasserstein距离](https://en.wikipedia.org/wiki/Wasserstein_metric)