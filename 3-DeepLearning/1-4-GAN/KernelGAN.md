## Blind Super-Resolution Kernel Estimation using an Internal-GAN

论文地址：https://arxiv.org/abs/1909.06581

作者：Sefi Bell-Kligler, Assaf Shocher, Michal Irani

机构：Dept. of Computer Science and Applied Math，The Weizmann Institute of Science, Israel

工程地址: http://www.wisdom.weizmann.ac.il/∼vision/kernelgan

代码： https://github.com/sefibk/KernelGAN

出版：NeurIPS 2019



### Internal-GAN 简介

论文地址：https://arxiv.org/abs/1812.00231

代码地址：https://github.com/assafshocher/InGAN

项目地址：http://www.wisdom.weizmann.ac.il/~vision/ingan/

发表：ICCV2019

Internal-GAN，在论文“InGAN: Capturing and Remapping the “DNA” of a Natural Image”中提出，它可以通过单张图像学习内部分布，从而生成不同尺寸、形状和比例的任意图像，整个过程就和俘获“DNA”一样。

​                           ![1581992224891](D:\Notes\raw_images\1581992224891.png)                       

如上图所示，红色框中是原始图像，其余均为InGAN生成。

#### 网络结构

![1581994031519](D:\Notes\raw_images\1581994031519.png)

对抗损失参考LSGAN：

   ![1581996561268](D:\Notes\raw_images\1581996561268.png)

重构损失参考CycleGAN：
$$
L_{reconst} = ||G(G(x;T);T^{-1}-x||_1
$$
InGAN的最终LOSS如下：
$$
L_{InGAN} = L_{GAN} + \lambda \cdot L_{reconst}
$$
![1581994521757](D:\Notes\raw_images\1581994521757.png)

生成器中，几何变换层（绿色）无参数，将特征图转换为期望的输出尺寸。下采样（蓝色）由卷积+最大池化完成，上采样（蓝色）由卷积+临近插值完成，瓶颈部（橙色）由6个残差网络块组成。

![1581994534200](D:\Notes\raw_images\1581994534200.png)

判别器中，采用了multiscale结构，scale_num即n如上图计算。对输入图片通过双线性插值，得到不同scale的新图片，送入同一种结构的判别器网络中，输出[0,1]得分。所有scale的得分加权，得到最终结果。

**训练细节：**采用Adam优化器和线性衰减学习率，batch_size为1，$L_{reconst}$ 的超参 λ 为0.1。每次迭代中，T的采样随机，从而导致不同的输出大小、形状和横纵比，变形范围随训练进行逐渐增大，直到覆盖所需区域。

**总结：**InGAN通过对单一图像的内部色块分布进行学习和分布匹配，在内部色块被学习到后可以实现图像的自然拉伸和重定向（可以产生大小，形状和纵横比明显不同的输出，包括非矩形输出图像）等任务，同时这个框架便可适用于这些任务而不需要再去更改。



### Kernel-GAN 简介   

如上式，对于SR问题，我们一般性假设，low-resolution(LR) 图片是由 high-resolution(HR)图片，经过一个理想SR-kernel（如双立方插值）降采样得到的。不同场景所对应的kernel各不相同，选对正确的 kernel，是获得好的结果的关键条件。我们的目标就是，给定一个 $I_{LR}$，通过某种方法，获得一个ks，从而推出 $I_{HR}$。

![1581996325655](D:\Notes\raw_images\1581996325655.png)

由此引出的问题是，如何根据一张特定的图片，获得其特有的SR-kernel。本文直接引用了另一篇论文中的结论：==“The correct SR-kernel is also the downscaling kernel which maximizes the similarity of patches across scales of the LR image”==。 [Michaeli & Irani: ICCV’2013]

**现有方法简述：**

1. Implicitly assume K *[RCAN, Zhang et al.; EDSR, Lim et al.]*

   缺点：对于通过另一个downscaling kernel降采样的$I_{LR}$，无法获得与源$I_{HR}$相似的结果。

2. Agnostic to K [PDN, Xianto et al; WDSR, Yu et al.]

3. Receive K as input [ZSSR, Shocher et al; SRMD, Zhang et al.]

#### 网络结构

![1581997707501](D:\Notes\raw_images\1581997707501.png)

Kernel GAN的损失函数：

   ![1581997769089](D:\Notes\raw_images\1581997769089.png)

G的作用是降采样，D的作用是判断，通过G生成的图片crop出的patch和原图crop出的patch的真和假，D输出的是一个0-1 D map，R为一个正则项。

#### 判别器

![1581997783514](D:\Notes\raw_images\1581997783514.png)

输入：$3*32*32$

卷积核：1个$7*7*3*64$，5个$1*1*64*64$，1个$1*1*64*1$

输出：$1*32*32$

#### 生成器

![1581997920601](D:\Notes\raw_images\1581997920601.png)

输入：$1*3*64*64 $ >>> $3*1*64*64$（swap_axis）

卷积核：1个$7*7*1*64$，1个$5*5*64*64$，1个$3*3*64*64$，2个$1*1*64*64$，1个$1*1*64*1$

最后一个卷积核的stride为2（downsampling）

输出：$3*1*32*32$

   ![1581998010991](D:\Notes\raw_images\1581998010991.png)

因为卷积和采样都是线性变换，所以把G设计为线性网络。但是，从经验来看，单层线性网络的效果并不好。论文给出一个猜想：在计算GAN Loss的时候，更新G部分的梯度时，此时的Loss函数为整个D网络，D部分包含非线性的运算。在只有一个有效解的问题中，从随机初始化的G的参数，更新到global minimum，几乎不可能。但是，多层线性网络，包含了许多较优的local minimum，更容易拟合到global minimum。

 **正则项：**

![1581998090880](D:\Notes\raw_images\1581998090880.png)

**训练细节:**  每张图片训练3000个迭代，G和D轮流训练，使用Adam优化器和step decay的学习率。

#### **总结**

1. 提出通过单张图片，找到与此图片相关的SR-Kernel的算法。

2. 完全无监督学习，在测试集上就可以获得结果。

3. 首次深度线性网络的实际应用。

