## 深度学习中的Lipschitz约束：泛化与生成模型

By 苏剑林 | 2018-10-07 | 20327位读者

文章地址：https://kexue.fm/archives/6051

前言：去年写过一篇WGAN-GP的入门读物[《互怼的艺术：从零直达WGAN-GP》](https://kexue.fm/archives/4439)，提到通过梯度惩罚来为WGAN的判别器增加Lipschitz约束（下面简称“L约束”）。前几天遐想时再次想到了WGAN，总觉得WGAN的梯度惩罚不够优雅，后来也听说WGAN在条件生成时很难搞（因为不同类的随机插值就开始乱了...），所以就想琢磨一下能不能搞出个新的方案来给判别器增加L约束。

闭门造车想了几天，然后发现想出来的东西别人都已经做了，果然是只有你想不到，没有别人做不到。主要包含在这两篇论文中：[《Spectral Norm Regularization for Improving the Generalizability of Deep Learning》](https://arxiv.org/abs/1705.10941)和[《Spectral Normalization for Generative Adversarial Networks》](https://arxiv.org/abs/1802.05957)。

所以这篇文章就按照自己的理解思路，对L约束相关的内容进行简单的介绍。注意本文的主题是L约束，并不只是WGAN。它可以用在生成模型中，也可以用在一般的监督学习中。

## L约束与泛化 #

### 扰动敏感 #

记输入为x，输出为y，模型为f，模型参数为w，记为

y=fw(x)   (1)
很多时候，我们希望得到一个“稳健”的模型。何为稳健？一般来说有两种含义，一是，比如模型变成了后是否还能达到相近的效果？如果在动力学系统中，还要考虑模型最终是否能恢复到；二是，比如输入从变成了后，是否能给出相近的预测结果。读者或许已经听说过深度学习模型存在“对抗攻击样本”，比如图片只改变一个像素就给出完全不一样的分类结果，这就是模型对输入过于敏感的案例。

### L约束[ #](https://kexue.fm/archives/6051#L约束)

所以，大多数时候我们都希望模型对输入扰动是不敏感的，这通常能提高模型的泛化性能。也就是说，我们希望∥x1−x2∥‖x1−x2‖很小时

∥fw(x1)−fw(x2)∥(2)(2)‖fw(x1)−fw(x2)‖


也尽可能地小。当然，“尽可能”究竟是怎样，谁也说不准。于是Lipschitz提出了一个更具体的约束，那就是存在某个常数（它只与参数有关，与输入无关），使得下式恒成立

∥fw(x1)−fw(x2)∥≤C(w)⋅∥x1−x2∥(3)(3)‖fw(x1)−fw(x2)‖≤C(w)⋅‖x1−x2‖


也就是说，希望整个模型被一个线性函数“控制”住。这便是了。

**换言之，在这里我们认为满足L约束的模型才是一个好模型～并且对于具体的模型，我们希望估算出C(w)C(w)的表达式，并且希望C(w)C(w)越小越好，越小意味着它对输入扰动越不敏感，泛化性越好。**

### 神经网络[ #](https://kexue.fm/archives/6051#神经网络)

在这里我们对具体的神经网络进行分析，以观察神经网络在什么时候会满足L约束。

简单其间，我们考虑单层的全连接f(Wx+b)f(Wx+b)，这里的ff是激活函数，而W,bW,b则是参数矩阵/向量，这时候[(3)](https://kexue.fm/archives/6051#mjx-eqn-eq%3Al-cond)(3)变为

∥f(Wx1+b)−f(Wx2+b)∥≤C(W,b)⋅∥x1−x2∥(4)(4)‖f(Wx1+b)−f(Wx2+b)‖≤C(W,b)⋅‖x1−x2‖


让充分接近，那么就可以将左边用一阶项近似，得到

∥∥∥∂f∂xW(x1−x2)∥∥∥≤C(W,b)⋅∥x1−x2∥(5)(5)‖∂f∂xW(x1−x2)‖≤C(W,b)⋅‖x1−x2‖


显然，要希望左边不超过右边，假定激活函数的梯度已经有界，尤其是我们常用的relu激活函数来说这个界还是1，因此这一项只带来一个常数，我们暂时忽略它，剩下来我们只需要考虑。

多层的神经网络可以逐步递归分析，从而最终还是单层的神经网络问题，而CNN、RNN等结构本质上还是特殊的全连接，所以照样可以用全连接的结果。因此，对于神经网络来说，问题变成了：如果

∥W(x1−x2)∥≤C∥x1−x2∥(6)(6)‖W(x1−x2)‖≤C‖x1−x2‖



恒成立，那么的值可以是多少？找出C的表达式后，我们就可以希望尽可能小，从而给参数带来一个正则化项。

## 矩阵范数 #

### 定义 #

其实到这里，我们已经将问题转化为了一个矩阵范数问题（矩阵范数的作用相当于向量的模长），它定义为

∥W∥2=maxx≠0∥Wx∥∥x∥(7)(7)‖W‖2=maxx≠0‖Wx‖‖x‖


如果是一个方阵，那么该范数又称为等，在本文中就算它不是方阵我们也叫它“谱范数（Spectral Norm）”好了。注意和都是指向量的范数，就是普通的向量模长。而左边的矩阵的范数我们本来没有明确定义的，但通过右边的向量模型的极限定义出来的，所以这类矩阵范数称为“由向量范数诱导出来的矩阵范数”。

好了，文绉绉的概念就不多说了，有了向量范数的概念之后，我们就有

∥W(x1−x2)∥≤∥W∥2⋅∥x1−x2∥(8)(8)‖W(x1−x2)‖≤‖W‖2⋅‖x1−x2‖


呃，其实也没做啥，就换了个记号而已，等于多少我们还是没有搞出来。

### Frobenius范数[ #](https://kexue.fm/archives/6051#Frobenius范数)

其实谱范数∥W∥2‖W‖2的准确概念和计算方法还是要用到比较多的线性代数的概念，我们暂时不研究它，而是先研究一个更加简单的范数：Frobenius范数，简称F范数。

这名字让人看着慌，其实定义特别简单，它就是

∥W∥F=∑i,jw2ij−−−−−−√(9)(9)‖W‖F=∑i,jwij2


说白了，它就是直接把矩阵当成一个向量，然后求向量的欧氏模长。

简单通过柯西不等式，我们就能证明

∥Wx∥≤∥W∥F⋅∥x∥(10)(10)‖Wx‖≤‖W‖F⋅‖x‖


很明显提供了的一个上界，也就是说，你可以理解为是式中最准确的（所有满足式的中最小的那个），但如果你不大关心精准度，你直接可以取，也能使得成立，毕竟容易计算。

### l2正则项[ #](https://kexue.fm/archives/6051#l2正则项)

前面已经说过，为了使神经网络尽可能好地满足L约束，我们应当希望C=∥W∥2C=‖W‖2尽可能小，我们可以把C2C2作为一个正则项加入到损失函数中。当然，我们还没有算出谱范数∥W∥2‖W‖2，但我们算出了一个更大的上界∥W∥F‖W‖F，那就先用着它吧，即loss为

loss=loss(y,fw(x))+λ∥W∥2F(11)(11)loss=loss(y,fw(x))+λ‖W‖F2


其中第一部分是指模型原来的loss。我们再来回顾一下的表达式，我们发现加入的正则项是

λ(∑i,jw2ij)(12)(12)λ(∑i,jwij2)


这不就是l2正则化吗？

终于，捣鼓了一番，我们得到了一点回报：我们揭示了l2正则化（也称为weight decay）与L约束的联系，表明l2正则化能使得模型更好地满足L约束，从而降低模型对输入扰动的敏感性，增强模型的泛化性能。

## 谱范数[ #](https://kexue.fm/archives/6051#谱范数)

### 主特征根[ #](https://kexue.fm/archives/6051#主特征根)

这部分我们来正式面对谱范数∥W∥2‖W‖2，这是线性代数的内容，比较理论化。

事实上，**谱范数∥W∥2‖W‖2等于W⊤WW⊤W的最大特征根（主特征根）的平方根，如果WW是方阵，那么∥W∥2‖W‖2等于WW的最大的特征根绝对值。**

> 注：对于感兴趣理论证明的读者，这里提供一下证明的大概思路。根据定义
>
> [(7)](https://kexue.fm/archives/6051#mjx-eqn-eq%3Am-norm)(7)
>
> 我们有
>
> ∥W∥22=maxx≠0x⊤W⊤Wxx⊤x=max∥x∥=1x⊤W⊤Wx‖W‖22=maxx≠0x⊤W⊤Wxx⊤x=max‖x‖=1x⊤W⊤Wx
>
> 假设
>
> W⊤WW⊤W
>
> 对角化为
>
> diag(λ1,…,λn)diag(λ1,…,λn)
>
> ，即
>
> W⊤W=U⊤diag(λ1,…,λn)UW⊤W=U⊤diag(λ1,…,λn)U
>
> ，其中
>
> λiλi
>
> 都是它的特征根，而且非负，而
>
> UU
>
> 是正交矩阵，由于正交矩阵与单位向量的积还是单位向量，那么
>
> ∥W∥22==≤=max∥x∥=1x⊤diag(λ1,…,λn)xmax∥x∥=1λ1x21+⋯+λnx2nmax{λ1,…,λn}(x21+⋯+x2n)(注意∥x∥=1）max{λ1,…,λn}‖W‖22=max‖x‖=1x⊤diag(λ1,…,λn)x=max‖x‖=1λ1x12+⋯+λnxn2≤max{λ1,…,λn}(x12+⋯+xn2)(注意‖x‖=1）=max{λ1,…,λn}
>
> 从而
>
> ∥W∥22‖W‖22
>
> 等于
>
> W⊤WW⊤W
>
> 的最大特征根。

### 幂迭代[ #](https://kexue.fm/archives/6051#幂迭代)

也许有读者开始不耐烦了：鬼愿意知道你是不是等于特征根呀，我关心的是怎么算这个鬼范数！！

事实上，前面的内容虽然看起来茫然，但却是求∥W∥2‖W‖2的基础。前一节告诉我们∥W∥22‖W‖22就是W⊤WW⊤W的最大特征根，所以问题变成了求W⊤WW⊤W的最大特征根，这可以通过[“幂迭代”法](https://en.wikipedia.org/wiki/Power_iteration)来解决。

所谓“幂迭代”，就是通过下面的迭代格式

u←(W⊤W)u∥(W⊤W)u∥(13)(13)u←(W⊤W)u‖(W⊤W)u‖


迭代若干次后，最后通过

∥W∥22≈u⊤W⊤Wu(14)(14)‖W‖22≈u⊤W⊤Wu


得到范数（也就是得到最大的特征根的近似值）。也可以等价改写为

v←W⊤u∥W⊤u∥,u←Wv∥Wv∥,∥W∥2≈u⊤Wv(15)(15)v←W⊤u‖W⊤u‖,u←Wv‖Wv‖,‖W‖2≈u⊤Wv


这样，初始化后（可以用全1向量初始化），就可以迭代若干次得到，然后代入算得的近似值。

> 注：对证明感兴趣的读者，这里照样提供一个简单的证明表明为什么这样的迭代会有效。
>
> 记A=W⊤WA=W⊤W，初始化为u(0)u(0)，同样假设AA可对角化，并且假设AA的各个特征根λ1,…,λnλ1,…,λn中，最大的特征根严格大于其余的特征根（不满足这个条件意味着最大的特征根是重根，讨论起来有点复杂，需要请读者查找专业证明，这里仅仅抛砖引玉。当然，从数值计算的角度，几乎没有两个人是完全相等的，因此可以认为重根的情况在实验中不会出现。），那么AA的各个特征向量η1,…,ηnη1,…,ηn构成完备的基底，所以我们可以设
>
> u(0)=c1η1+⋯+cnηnu(0)=c1η1+⋯+cnηn
>
> 每次的迭代是
>
> Au/∥Au∥Au/‖Au‖
>
> ，其中分母只改变模长，我们留到最后再执行，只看
>
> AA
>
> 的重复作用
>
> Aru(0)=c1Arη1+⋯+cnArηnAru(0)=c1Arη1+⋯+cnArηn
>
> 注意对于特征向量有
>
> Aη=ληAη=λη
>
> ，从而
>
> Aru(0)=c1λr1η1+⋯+cnλrnηnAru(0)=c1λ1rη1+⋯+cnλnrηn
>
> 不失一般性设
>
> λ1λ1
>
> 为最大的特征值，那么
>
> Aru(0)λr1=c1η1+c2(λ2λ1)r+⋯+cn(λnλ1)rηnAru(0)λ1r=c1η1+c2(λ2λ1)r+⋯+cn(λnλ1)rηn
>
> 根据假设
>
> λ2/λ1,…,λn/λ1λ2/λ1,…,λn/λ1
>
> 都小于1，所以
>
> r→∞r→∞
>
> 时它们都趋于零，或者说当
>
> rr
>
> 足够大时它们可以忽略，那么就有
>
> Aru(0)λr1≈c1η1Aru(0)λ1r≈c1η1
>
> 先不管模长，这个结果表明当
>
> rr
>
> 足够大时，
>
> Aru(0)Aru(0)
>
> 提供了最大的特征根对应的特征向量的近似方向，其实每一步的归一化只是为了防止溢出而已。这样一来
>
> u=Aru(0)/∥Aru(0)∥u=Aru(0)/‖Aru(0)‖
>
> 就是对应的单位特征向量，即
>
> Au=λ1uAu=λ1u
>
> 因此
>
> u⊤Au=λ1u⊤u=λ1u⊤Au=λ1u⊤u=λ1
>
> 这就求出了谱范数的平方。

### 谱正则化[ #](https://kexue.fm/archives/6051#谱正则化)

前面我们已经表明了Frobenius范数与l2正则化的关系，而我们已经说明了Frobenius范数是一个更强（更粗糙）的条件，更准确的范数应该是谱范数。虽然谱范数没有Frobenius范数那么容易计算，但依然可以通过式[(15)](https://kexue.fm/archives/6051#mjx-eqn-eq%3Am-norm-iter)(15)迭代几步来做近似。

所以，我们可以提出“谱正则化（Spectral Norm Regularization）”的概念，即把谱范数的平方作为额外的正则项，取代简单的l2正则项。即式[(11)](https://kexue.fm/archives/6051#mjx-eqn-eq%3Al2-regular)(11)变为

loss=loss(y,fw(x))+λ∥W∥22(16)(16)loss=loss(y,fw(x))+λ‖W‖22



[《Spectral Norm Regularization for Improving the Generalizability of Deep Learning》](https://arxiv.org/abs/1705.10941)一文已经做了多个实验，表明“谱正则化”在多个任务上都能提升模型性能。

在Keras中，可以通过下述代码计算谱范数

```python
def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))
```

## 生成模型[ #](https://kexue.fm/archives/6051#生成模型)

### WGAN[ #](https://kexue.fm/archives/6051#WGAN)

如果说在普通的监督训练模型中，L约束只是起到了“锦上添花”的作用，那么在WGAN的判别器中，L约束就是必不可少的关键一步了。因为WGAN的判别器的优化目标是

W(Pr,Pg)=sup|f|L=1Ex∼Pr[f(x)]−Ex∼Pg[f(x)](17)(17)W(Pr,Pg)=sup|f|L=1Ex∼Pr[f(x)]−Ex∼Pg[f(x)]


这里的分别是真实分布和生成分布，指的就是要满足特定的L约束（那个）。所以上述目标的意思是，在所有满足这个L约束的函数中，挑出使得最大的那个，就是最理想的判别器。写成loss的形式就是

min|f|L=1Ex∼Pg[f(x)]−Ex∼Pr[f(x)](18)(18)min|f|L=1Ex∼Pg[f(x)]−Ex∼Pr[f(x)]



### 梯度惩罚[ #](https://kexue.fm/archives/6051#梯度惩罚)

目前比较有效的一种方案就是梯度惩罚，即∥f′(x)∥=1‖f′(x)‖=1是|f|L=1|f|L=1的一个充分条件，那么我把这一项加入到判别器的loss中作为惩罚项，即

minfEx∼Pg[f(x)]−Ex∼Pr[f(x)]＋λ(∥f′(xinter)∥−1)2(19)(19)minfEx∼Pg[f(x)]−Ex∼Pr[f(x)]＋λ(‖f′(xinter)‖−1)2


事实上我觉得加个会更好

minfEx∼Pg[f(x)]−Ex∼Pr[f(x)]＋λmax(∥f′(xinter)∥−1,0)2(20)(20)minfEx∼Pg[f(x)]−Ex∼Pr[f(x)]＋λmax(‖f′(xinter)‖−1,0)2


其中采用随机插值的方式

xinter=εxreal+(1−ε)xfakeε∼U[0,1],xreal∼Pr,xfake∼Pg(21)(21)xinter=εxreal+(1−ε)xfakeε∼U[0,1],xreal∼Pr,xfake∼Pg


梯度惩罚不能保证，但是直觉上它会在1附近浮动，所以理论上也在1附近浮动，从而近似达到L约束。

这种方案在很多情况下都已经work得比较好了，但是在真实样本的类别数比较多的时候却比较差（尤其是条件生成）。问题就出在随机插值上：原则上来说，L约束要在整个空间满足才行，但是通过线性插值的梯度惩罚只能保证在一小块空间满足。如果这一小块空间刚好差不多就是真实样本和生成样本之间的空间，那勉勉强强也就够用了，但是如果类别数比较多，不同的类别进行插值，往往不知道插到哪里去了，导致该满足L条件的地方不满足，因此判别器就失灵了。

> 思考：梯度惩罚能不能直接用作有监督的模型的正则项呢？有兴趣的读者可以试验一下～

### 谱归一化[ #](https://kexue.fm/archives/6051#谱归一化)

梯度惩罚的问题在于它只是一个惩罚，只能在局部生效。真正妙的方案是构造法：构建特殊的ff，使得不管ff里边的参数是什么，ff都满足L约束。

事实上，WGAN首次提出时用的是参数裁剪——将所有参数的绝对值裁剪到不超过某个常数，这样一来参数的Frobenius范数不会超过某个常数，从而|f|L|f|L不会超过某个常数，虽然没有准确地实现|f|L=1|f|L=1，但这只会让loss放大常数倍，因此不影响优化结果。参数裁剪就是一种构造法，这不过这种构造法对优化并不友好。

简单来看，这种裁剪的方案优化空间有很大，比如改为将所有参数的Frobenius范数裁剪到不超过某个常数，这样模型的灵活性比直接参数裁剪要好。如果觉得裁剪太粗暴，换成参数惩罚也是可以的，即对所有范数超过Frobenius范数的参数施加一个大惩罚，我也试验过，基本有效，但是收敛速度比较慢。

然而，上面这些方案都只是某种近似，现在我们已经有了谱范数，那么可以用最精准的方案了：**将ff中所有的参数都替换为w/∥w∥2w/‖w‖2**。这就是谱归一化（Spectral Normalization），在[《Spectral Normalization for Generative Adversarial Networks》](https://arxiv.org/abs/1802.05957)一文中被提出并实验。这样一来，如果ff所用的激活函数的导数绝对值都不超过1，那么我们就有|f|L≤1|f|L≤1，从而用最精准的方案实现了所需要的L约束。

> 注：“激活函数的导数绝对值都不超过1”，这个通常都能满足，但是如果判别模型使用了残差结构，则激活函数相当于是x+relu(Wx+b)x+relu(Wx+b)，这时候它的导数就不一定不超过1了。但不管怎样，它会不超过一个常数，因此不影响优化结果。

我自己尝试过在WGAN中使用谱归一化（不加梯度惩罚，参考代码见后面），发现最终的收敛速度（达到同样效果所需要的epoch）比WGAN-GP还要快，效果还要更好一些。而且，还有一个影响速度的原因：就是每个epoch的运行时间，梯度惩罚会比用谱归一化要长，因为用了梯度惩罚后，在梯度下降的时候相当于要算二次梯度了，要执行整个前向过程两次，所以速度比较慢。

### Keras实现[ #](https://kexue.fm/archives/6051#Keras实现)

在Keras中，实现谱归一化可以说简单也可以说不简单。

说简单，只需要在判别器的每一层卷积层和全连接层都传入kernel_constraint参数，而BN层传入gamma_constraint参数。constraint的写法是

```python
def spectral_normalization(w):
    return w / spectral_norm(w)
```

参考代码：
https://github.com/bojone/gan/blob/master/keras/wgan_sn_celeba.py

说不简单，是因为目前的Keras（2.2.4版本）中的kernel_constraint并没有真正改变了kernel，而只是在梯度下降之后对kernel的值进行了调整，这跟论文中spectral_normalization的方式并不一样。如果只是这样使用的话，就会发现后期的梯度不准，模型的生成质量不佳。为了实现真正地修改kernel，我们要不就得重新定义所有的层（卷积、全连接、BN等所有包含矩阵乘法的层），要不就只能修改源码了，修改源码是最简单的方案，修改文件keras/engine/base_layer.py的Layer对象的add_weight方法，本来是（目前是222行开始）：

```python
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.
        # Returns
            The created weight variable.
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
        if regularizer is not None:
            with K.name_scope('weight_regularizer'):
                self.add_loss(regularizer(weight))
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight
```

修改为

```python
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.
        # Returns
            The created weight variable.
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=None)
        if regularizer is not None:
            with K.name_scope('weight_regularizer'):
                self.add_loss(regularizer(weight))
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        if constraint is not None:
            return constraint(weight)
        return weight
```

也就是把K.variable的constraint改为None，把constraint放到最后执行～**注意，不要看到要改源码就马上来吐槽Keras封装太死，不够灵活什么的，你要是用其他框架基本上比Keras复杂好多倍（相对不加spectral_normalization的GAN的改动量）。**

（更新：一个新的不用修改源码的实现方式在[这里](https://kexue.fm/archives/6311/)。）



[![修改源码前的wgan-sn结果](https://kexue.fm/usr/uploads/2018/10/4268425315.png)](https://kexue.fm/usr/uploads/2018/10/4268425315.png)

修改源码前的wgan-sn结果

[![修改源码后的wgan-sn结果](https://kexue.fm/usr/uploads/2018/10/1411501175.png)](https://kexue.fm/usr/uploads/2018/10/1411501175.png)

修改源码后的wgan-sn结果



## 总结[ #](https://kexue.fm/archives/6051#总结)

本文是关于Lipschitz约束的一篇总结，主要介绍了如何使得模型更好地满足Lipschitz约束，这关系到模型的泛化能力。而难度比较大的概念是谱范数，涉及较多的理论和公式。

整体来看，关于谱范数的相关内容都是比较精巧的，而相关结论也进一步表明线性代数跟机器学习紧密相关，很多“高深”的线性代数内容都可以在机器学习中找到对应的应用。

***转载到请包括本文地址：**https://kexue.fm/archives/6051*

***更详细的转载事宜请参考：***[《科学空间FAQ》](https://kexue.fm/archives/6508#文章如何转载/引用)

**如果您需要引用本文，请参考：**

苏剑林. (2018, Oct 07). 《深度学习中的Lipschitz约束：泛化与生成模型 》[Blog post]. Retrieved from https://kexue.fm/archives/6051