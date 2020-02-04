## A Simple yet Effective Way for Improving the Performance of GANs

论文地址：https://arxiv.org/abs/1911.10979

作者：Yong-Goo Shin, Yoon-Jae Yeo, Min-Cheol Sagong, Cheol-Hwan Yoo, Sung-Jea Ko

机构：Korea university

文章地址：https://kexue.fm/archives/7105



摘要

> This paper presents a simple but effective way that improves the performance of generative adversarial networks (GANs) without imposing the training overhead or modifying the network architectures of existing methods. The proposed method employs a novel cascading rejection (CR) module for discriminator, which extracts multiple non-overlapped features in an iterative manner. The CR module supports the discriminator to effectively distinguish between real and generated images, which results in a strong penalization to the generator. In order to deceive the robust discriminator containing the CR module, the generator produces the images that are more similar to the real images. Since the proposed CR module requires only a few simple vector operations, it can be readily applied to existing frameworks with marginal training overheads. Quantitative evaluations on various datasets including CIFAR-10, Celeb-HQ, LSUN, and tiny-ImageNet confirm that the proposed method significantly improves the performance of GANs and conditional GANs in terms of Frechet inception distance (FID) indicating the diversity and visual appearance of the generated images.



昨天刷arxiv时发现了一篇来自~~星星~~韩国的论文，名字很直白，就叫做[《A Simple yet Effective Way for Improving the Performance of GANs》](https://arxiv.org/abs/1911.10979)。打开一看，发现内容也很简练，就是提出了一种加强GAN的判别器的方法，能让GAN的生成指标有一定的提升。

作者把这个方法叫做Cascading Rejection，我不知道咋翻译，扔到百度翻译里边显示“级联抑制”，想想看好像是有这么点味道，就暂时这样叫着了。介绍这个方法倒不是因为它有多强大，而是觉得它的几何意义很有趣，而且似乎有一定的启发性。

## 正交分解[ #](https://kexue.fm/archives/7105#正交分解)

GAN的判别器一般是经过多层卷积后，通过flatten或pool得到一个固定长度的向量vv，然后再与一个权重向量ww做内积，得到一个标量打分（先不考虑偏置项和激活函数等末节）：

D(x)=⟨v,w⟩(1)(1)D(x)=⟨v,w⟩


也就是说，用作为输入图片的表征，然后通过和的内积大小来判断出这个图片的“真”的程度。

然而，⟨v,w⟩⟨v,w⟩只取决于vv在ww上的投影分量，换言之，固定⟨v,w⟩⟨v,w⟩和ww时，vv仍然可以有很大的变动，如下面左图所示。

[![与w内积相等的v向量可以差异很大](https://kexue.fm/usr/uploads/2019/11/873030710.png)](https://kexue.fm/usr/uploads/2019/11/873030710.png)

与w内积相等的v向量可以差异很大

[![v的投影分量和垂直分量](https://kexue.fm/usr/uploads/2019/11/361353465.png)](https://kexue.fm/usr/uploads/2019/11/361353465.png)

v的投影分量和垂直分量



假如我们认为⟨v,w⟩⟨v,w⟩等于某个值时图片就为真，问题是vv变化那么大，难道每一个vv都代表一张真实图片吗？显然不一定。这就反映了通过内积来打分的问题所在：它只考虑了在ww上的投影分量，没有考虑垂直分量（如上面右图）：

v−∥v∥cos(v,w)w∥w∥=v−⟨v,w⟩∥w∥2w(2)(2)v−‖v‖cos⁡(v,w)w‖w‖=v−⟨v,w⟩‖w‖2w



既然如此，一个很自然的想法是：能否用另一个参数向量来对这个垂直分量在做一次分类呢？显然是可以的，而且这个垂直分量的再次分类时也会导致一个新的垂直分量，因此这个过程可以迭代下去：

⎧⎩⎨⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪v1=vD1(x)=⟨v1,w1⟩v2=v1−⟨v1,w1⟩∥w1∥2w1D2(x)=⟨v2,w2⟩v3=v2−⟨v2,w2⟩∥w2∥2w2D3(x)=⟨v3,w3⟩v4=v3−⟨v3,w3⟩∥w3∥2w3⋮DN(x)=⟨vN,wN⟩(3)(3){v1=vD1(x)=⟨v1,w1⟩v2=v1−⟨v1,w1⟩‖w1‖2w1D2(x)=⟨v2,w2⟩v3=v2−⟨v2,w2⟩‖w2‖2w2D3(x)=⟨v3,w3⟩v4=v3−⟨v3,w3⟩‖w3‖2w3⋮DN(x)=⟨vN,wN⟩



## 分析思考[ #](https://kexue.fm/archives/7105#分析思考)

其实写到这，原论文的思路基本上已经说完了，剩下的是一些细节上的操作。首先已经有了NN个打分D1(x),D2(x),…,DN(x)D1(x),D2(x),…,DN(x)，每个打分都可以应用判别器的loss（直接用hinge loss或者加sigmoid激活后用交叉熵），最后对这NN个loss加权平均，作为最终的判别器loss，仅这样就能带来GAN的性能提升了。作者还将其进一步推广到CGAN中，也得到了不错的效果。



[![论文提出的GAN技巧的实验结果](https://kexue.fm/usr/uploads/2019/11/1864282304.png)](https://kexue.fm/usr/uploads/2019/11/1864282304.png)

论文提出的GAN技巧的实验结果



相比实验结果，笔者认为这个技巧更深层次的意义更值得关注。其实这个思路可以按理说可以用到一般的分类问题中而不单单是GAN。由于把垂直分量都迭代地加入了预测，我们可以认为参数w1,w2,…,wNw1,w2,…,wN分别代表了NN个不同的视角，而每一个分类相当于在不同的视角下进行分类判断。

想到这里，笔者想起了Hinton的[Capsule](https://kexue.fm/tag/Capsule/)。虽然形式上不大一样，但本意上似乎有相通之处，Capsule希望用一个向量而不是标量来表示一个实体，这里的“级联抑制”也是通过不断进行垂直分解来给出多个角度的分类结果，也就是说认定一个向量是不是属于一个类，必须给出多个打分而不单是一个，这也有“用向量而不是标量”的味道。

遗憾的是，笔者按上述思路简单实验了一下（cifar10），发现验证集的分类准确率下降了一点（注意这跟GAN的结果不矛盾，提升GAN的表现是因为加大了判别难度，但是有监督的分类模型不希望加大判别难度），但是好在过拟合程度也减少了（即训练集和验证集的准确率差距减少了），当然笔者的实验过于简陋，不能做到严谨地下结论。不过笔者依然觉得，由于其鲜明的几何意义，这个技巧仍然值得进一步思考。

## 文章小结[ #](https://kexue.fm/archives/7105#文章小结)

本文介绍了一个具有鲜明几何意义的提升GAN表现的技巧，并且进一步讨论了它进一步的潜在价值。

***转载到请包括本文地址：**https://kexue.fm/archives/7105*

***更详细的转载事宜请参考：***[《科学空间FAQ》](https://kexue.fm/archives/6508#文章如何转载/引用)