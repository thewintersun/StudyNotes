## Wasserstein Gradient Penalty

论文地址：https://arxiv.org/abs/1704.00028 

作者：Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville

机构：Montreal Institute for Learning Algorithms

发表：NIPS 2017

Tensorflow实现：https://github.com/igul222/improved_wgan_training 



文章来源：https://www.zhihu.com/question/52602529/answer/158727900

作者：郑华滨  时间：2017-04-21

前段时间，Wasserstein  GAN以其精巧的理论分析、简单至极的算法实现、出色的实验效果，在GAN研究圈内掀起了一阵热潮。但是很多人（包括我们实验室的同学）到了上手跑实验的时候，却发现WGAN实际上没那么完美，反而存在着训练困难、收敛速度慢等问题。其实，[WGAN的作者Martin Arjovsky不久后就在reddit上表示他也意识到了这个问题](https://link.zhihu.com/?target=https%3A//www.reddit.com/r/MachineLearning/comments/5zd4c0/d_survey_whats_the_most_stable_regiment_for/dexfhxu/%3Futm_content%3Dpermalink%26utm_medium%3Dfront%26utm_source%3Dreddit%26utm_name%3DMachineLearning)，认为关键在于原设计中Lipschitz限制的施加方式不对，并在新论文中提出了相应的改进方案：

**首先回顾一下WGAN的关键部分——Lipschitz限制是什么。**WGAN中，判别器D和生成器G的loss函数分别是：

![[公式]](https://www.zhihu.com/equation?tex=L%28D%29+%3D+-%5Cmathbb%7BE%7D_%7Bx%5Csim+P_r%7D%5BD%28x%29%5D+%2B+%5Cmathbb%7BE%7D_%7Bx%5Csim+P_g%7D%5BD%28x%29%5D) （公式1）

![[公式]](https://www.zhihu.com/equation?tex=L%28G%29+%3D+-+%5Cmathbb%7BE%7D_%7Bx%5Csim+P_g%7D%5BD%28x%29%5D)  （公式2）

公式1表示判别器希望尽可能拉高真样本的分数，拉低假样本的分数，公式2表示生成器希望尽可能拉高假样本的分数。

Lipschitz限制则体现为，在整个样本空间 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BX%7D)  上，要求判别器函数D(x)梯度的Lp-norm不大于一个有限的常数K：

![[公式]](https://www.zhihu.com/equation?tex=%7C%7C+%5Cnabla+_x+D%28x%29+%7C%7C_p+%5Cleq+K+%5Ctext%7B%2C+%7D+%5Cforall+x+%5Cin+%5Cmathcal%7BX%7D)  （公式3）

直观上解释，就是当输入的样本稍微变化后，判别器给出的分数不能发生太过剧烈的变化。在原来的论文中，这个限制具体是通过weight clipping的方式实现的：每当更新完一次判别器的参数之后，就检查判别器的所有参数的绝对值有没有超过一个阈值，比如0.01，有的话就把这些参数clip回 [-0.01, 0.01] 范围内。通过在训练过程中保证判别器的所有参数有界，就保证了判别器不能对两个略微不同的样本给出天差地别的分数值，从而间接实现了Lipschitz限制。

**然而weight clipping的实现方式存在两个严重问题：**

第一，如公式1所言，==判别器loss希望尽可能拉大真假样本的分数差，然而weight clipping独立地限制每一个网络参数的取值范围，在这种情况下我们可以想象，最优的策略就是尽可能让所有参数走极端，要么取最大值（如0.01）要么取最小值（如-0.01）！== 为了验证这一点，作者统计了经过充分训练的判别器中所有网络参数的数值分布，发现真的集中在最大和最小两个极端上：

![img](https://pic3.zhimg.com/50/v2-7a3aedf9fa60ce660bff9f03935d8f15_hd.jpg)

这样带来的结果就是，判别器会非常倾向于学习一个简单的映射函数（想想看，几乎所有参数都是正负0.01，都已经可以直接视为一个二值神经网络了，太简单了）。而作为一个深层神经网络来说，这实在是对自身强大拟合能力的巨大浪费！判别器没能充分利用自身的模型能力，经过它回传给生成器的梯度也会跟着变差。

在正式介绍gradient penalty之前，我们可以先看看在它的指导下，同样充分训练判别器之后，参数的数值分布就合理得多了，判别器也能够充分利用自身模型的拟合能力：

![img](https://pic3.zhimg.com/50/v2-27afb895eea82f5392b19ca770865b96_hd.jpg)

第二个问题，==weight clipping会导致很容易一不小心就梯度消失或者梯度爆炸==。原因是判别器是一个多层网络，如果我们把clipping threshold设得稍微小了一点，每经过一层网络，梯度就变小一点点，多层之后就会指数衰减；反之，如果设得稍微大了一点，每经过一层网络，梯度变大一点点，多层之后就会指数爆炸。==只有设得不大不小，才能让生成器获得恰到好处的回传梯度，然而在实际应用中这个平衡区域可能很狭窄，就会给调参工作带来麻烦==。相比之下，gradient penalty就可以让梯度在后向传播的过程中保持平稳。论文通过下图体现了这一点，其中横轴代表判别器从低到高第几层，纵轴代表梯度回传到这一层之后的尺度大小（注意纵轴是对数刻度），c是clipping threshold：

![img](https://pic4.zhimg.com/50/v2-34114a10c56518d606c1b5dd77f64585_hd.jpg)

**说了这么多，gradient penalty到底是什么？**

前面提到，Lipschitz限制是要求判别器的梯度不超过K，那我们何不直接设置一个额外的loss项来体现这一点呢？比如说： 

![[公式]](https://www.zhihu.com/equation?tex=ReLU%5B+%7C%7C+%5Cnabla_x+D%28x%29+%7C%7C_p+-+K+%5D)  （公式4）

不过，==既然判别器希望尽可能拉大真假样本的分数差距，那自然是希望梯度越大越好，变化幅度越大越好，所以判别器在充分训练之后，其梯度norm其实就会是在K附近==。知道了这一点，我们可以把上面的loss改成要求梯度norm离K越近越好，效果是类似的：

![[公式]](https://www.zhihu.com/equation?tex=+%5B+%7C%7C+%5Cnabla_x+D%28x%29+%7C%7C_p+-+K+%5D%5E2)  （公式5）

究竟是公式4好还是公式5好，我看不出来，可能需要实验验证，反正论文作者选的是公式5。接着我们简单地把K定为1，再跟WGAN原来的判别器loss加权合并，就得到新的判别器loss：

![[公式]](https://www.zhihu.com/equation?tex=L%28D%29+%3D+-%5Cmathbb%7BE%7D_%7Bx%5Csim+P_r%7D%5BD%28x%29%5D+%2B+%5Cmathbb%7BE%7D_%7Bx%5Csim+P_g%7D%5BD%28x%29%5D+%2B+%5Clambda+%5Cmathbb%7BE%7D_%7Bx+%5Csim+%5Cmathcal%7BX%7D%7D+%5B+%7C%7C+%5Cnabla_x+D%28x%29+%7C%7C_p+-+1+%5D%5E2)  （公式6）

这就是所谓的Gradient Penalty了吗？还没完。公式6有两个问题，==首先是loss函数中存在梯度项，那么优化这个loss岂不是要算梯度的梯度==？一些读者可能对此存在疑惑，不过这属于实现上的问题，放到后面说。

其次，3个loss项都是期望的形式，落到实现上肯定得变成采样的形式。前面两个期望的采样我们都熟悉，第一个期望是从真样本集里面采，第二个期望是从生成器的噪声输入分布采样后，再由生成器映射到样本空间。可是第三个分布要求我们在整个样本空间 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BX%7D) 上采样，这完全不科学！由于所谓的维度灾难问题，如果要通过采样的方式在图片或自然语言这样的高维样本空间中估计期望值，所需样本量是指数级的，实际上没法做到。

所以，论文作者就非常机智地提出，==我们其实没必要在整个样本空间上施加Lipschitz限制，只要重点抓住生成样本集中区域、真实样本集中区域以及夹在它们中间的区域就行了==。具体来说，我们先随机采一对真假样本，还有一个0-1的随机数：

![[公式]](https://www.zhihu.com/equation?tex=x_r+%5Csim+P_r%2C+x_g+%5Csim+P_g%2C+%5Cepsilon+%5Csim+Uniform%5B0%2C+1%5D) （公式7）

然后在 ![[公式]](https://www.zhihu.com/equation?tex=x_r) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_g) 的连线上随机插值采样：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat+x+%3D+%5Cepsilon+x_r+%2B+%281+-+%5Cepsilon%29+x_g)  （公式8）

把按照上述流程采样得到的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+x) 所满足的分布记为 ![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5Chat+x%7D) ，就得到最终版本的判别器loss：

![[公式]](https://www.zhihu.com/equation?tex=L%28D%29+%3D+-%5Cmathbb%7BE%7D_%7Bx%5Csim+P_r%7D%5BD%28x%29%5D+%2B+%5Cmathbb%7BE%7D_%7Bx%5Csim+P_g%7D%5BD%28x%29%5D+%2B+%5Clambda+%5Cmathbb%7BE%7D_%7Bx+%5Csim+%5Cmathcal%7BP_%7B%5Chat+x%7D%7D%7D+%5B+%7C%7C+%5Cnabla_x+D%28x%29+%7C%7C_p+-+1+%5D%5E2)  （公式9）

**这就是新论文所采用的gradient penalty方法，相应的新WGAN模型简称为WGAN-GP。**我们可以做一个对比：

- weight clipping是对样本空间全局生效，但因为是间接限制判别器的梯度norm，会导致一不小心就梯度消失或者梯度爆炸；
- gradient penalty只对真假样本集中区域、及其中间的过渡地带生效，但因为是直接把判别器的梯度norm限制在1附近，所以梯度可控性非常强，容易调整到合适的尺度大小。

论文还讲了一些使用gradient penalty时需要注意的配套事项，==这里只提一点：由于我们是对每个样本独立地施加梯度惩罚，所以判别器的模型架构中不能使用Batch Normalization，因为它会引入同个batch中不同样本的相互依赖关系===。如果需要的话，可以选择其他normalization方法，如Layer Normalization、Weight Normalization和Instance Normalization，这些方法就不会引入样本之间的依赖。论文推荐的是Layer Normalization。

实验表明，gradient penalty能够显著提高训练速度，解决了原始WGAN收敛缓慢的问题：

![img](https://pic1.zhimg.com/50/v2-5b01ef93f60a14e7fa10dbea2b620627_hd.jpg)

虽然还是比不过DCGAN，但是因为WGAN不存在平衡判别器与生成器的问题，所以会比DCGAN更稳定，还是很有优势的。不过，作者凭什么能这么说？因为下面的实验体现出，在各种不同的网络架构下，其他GAN变种能不能训练好，可以说是一件相当看人品的事情，但是WGAN-GP全都能够训练好，尤其是最下面一行所对应的101层残差神经网络：

![img](https://pic4.zhimg.com/50/v2-e0a3d86ccfa101a4d3fee1c0cef96a81_hd.jpg)

**剩下的实验结果中，比较厉害的是第一次成功做到了“纯粹的”的文本GAN训练！**我们知道在图像上训练GAN是不需要额外的有监督信息的，但是之前就没有人能够像训练图像GAN一样训练好一个文本GAN，要么依赖于预训练一个语言模型，要么就是利用已有的有监督ground truth提供指导信息。而现在WGAN-GP终于在无需任何有监督信息的情况下，生成出下图所示的英文字符序列：

![img](https://pic1.zhimg.com/50/v2-33c3af150f9bd52485b800948d3cb700_hd.jpg)

它是怎么做到的呢？我认为关键之处是对样本形式的更改。**以前我们一般会把文本这样的离散序列样本表示为sequence of index，但是它把文本表示成sequence of probability vector。**对于生成样本来说，我们可以取网络softmax层输出的词典概率分布向量，作为序列中每一个位置的内容；而对于真实样本来说，每个probability vector实际上就蜕化为我们熟悉的onehot vector。

但是如果按照传统GAN的思路来分析，这不是作死吗？一边是hard onehot vector，另一边是soft probability vector，判别器一下子就能够区分它们，生成器还怎么学习？没关系，对于WGAN来说，真假样本好不好区分并不是问题，WGAN只是拉近两个分布之间的Wasserstein距离，就算是一边是hard onehot另一边是soft probability也可以拉近，在训练过程中，概率向量中的有些项可能会慢慢变成0.8、0.9到接近1，整个向量也会接近onehot，最后我们要真正输出sequence of index形式的样本时，只需要对这些概率向量取argmax得到最大概率的index就行了。

新的样本表示形式+WGAN的分布拉近能力是一个“黄金组合”，但除此之外，还有其他因素帮助论文作者跑出上图的效果，包括：

- 文本粒度为英文字符，而非英文单词，所以字典大小才二三十，大大减小了搜索空间。
- 文本长度也才32.
- 生成器用的不是常见的LSTM架构，而是多层反卷积网络，输入一个高斯噪声向量，直接一次性转换出所有32个字符。

**最后说回gradient penalty的实现问题。**==loss中本身包含梯度，优化loss就需要求梯度的梯度，这个功能并不是现在所有深度学习框架的标配功能==，不过好在Tensorflow就有提供这个接口——tf.gradients。开头链接的GitHub源码中就是这么写的：

```python
# interpolates就是随机插值采样得到的图像，gradients就是loss中的梯度惩罚项
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
```

对于我这样的PyTorch党就非常不幸了，高阶梯度的功能还在开发，感兴趣的PyTorch党可以订阅这个GitHub的pull request：Autograd refactor，如果它被merged了话就可以在最新版中使用高阶梯度的功能实现gradient penalty了。  但是除了等待我们就没有别的办法了吗？**其实可能是有的，我想到了一种近似方法来实现gradient penalty，只需要把微分换成差分：**

![[公式]](https://www.zhihu.com/equation?tex=L%28D%29+%3D+-%5Cmathbb%7BE%7D_%7Bx%5Csim+P_r%7D%5BD%28x%29%5D+%2B+%5Cmathbb%7BE%7D_%7Bx%5Csim+P_g%7D%5BD%28x%29%5D+%2B+%5Clambda+%5Cmathbb%7BE%7D_%7Bx_1+%5Csim+%5Cmathcal%7BP_%7B%5Chat+x%7D%7D%2C+%5C+x_2+%5Csim+%5Cmathcal%7BP_%7B%5Chat+x%7D%7D%7D+%5B+%5Cfrac%7B%7CD%28x_1%29+-D%28x_2%29%7C%7D%7B+%7C%7C+x_1+-+x_2+%7C%7C_p+%7D+-+1%5D%5E2)  

也就是说，我们仍然是在分布 ![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5Chat+x%7D) 上随机采样，但是一次采两个，然后要求它们的连线斜率要接近1，这样理论上也可以起到跟公式9一样的效果，我自己在MNIST+MLP上简单验证过有作用，PyTorch党甚至Tensorflow党都可以尝试用一下。

