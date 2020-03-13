诺亚项目地址：https://github.com/huawei-noah



南大图像压缩论文： https://arxiv.org/abs/1805.11394

[Yiming Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+Y), [Siyang Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+S), [Jianquan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Xingang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Qingyi Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+Q)

> In recent years, deep neural networks have achieved great success in the field of computer vision. However, it is still a big challenge to deploy these deep models on resource-constrained embedded devices such as mobile robots, smart phones and so on. Therefore, network compression for such platforms is a reasonable solution to reduce memory consumption and computation complexity. In this paper, a novel channel pruning method based on genetic algorithm is proposed to compress very deep Convolution Neural Networks (CNNs). Firstly, a pre-trained CNN model is pruned layer by layer according to the sensitivity of each layer. After that, the pruned model is fine-tuned based on knowledge distillation framework. These two improvements significantly decrease the model redundancy with less accuracy drop. Channel selection is a combinatorial optimization problem that has exponential solution space. In order to accelerate the selection process, the proposed method formulates it as a search problem, which can be solved efficiently by genetic algorithm. Meanwhile, a two-step approximation fitness function is designed to further improve the efficiency of genetic process. The proposed method has been verified on three benchmark datasets with two popular CNN models: VGGNet and ResNet. On the CIFAR-100 and ImageNet datasets, our approach outperforms several state-of-the-art methods. On the CIFAR-10 and SVHN datasets, the pruned VGGNet achieves better performance than the original model with 8 times parameters compression and 3 times FLOPs reduction.



CARS: Continuous Evolution for Efficient Neural Architecture Search

 https://arxiv.org/pdf/1909.04977.pdf



## Data-Free Learning of Student Networks

http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Data-Free_Learning_of_Student_Networks_ICCV_2019_paper.html

代码地址：https://github.com/huawei-noah

**Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, Qi Tian**; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 3514-3522

Learning portable neural networks is very essential for computer vision for the purpose that pre-trained heavy deep models can be well applied on edge devices such as mobile phones and micro sensors. Most existing deep neural network compression and speed-up methods are very effective for training compact deep models, when we can directly access the training dataset. However, training data for the given deep network are often unavailable due to some practice problems (e.g. privacy, legal issue, and transmission), and the architecture of the given network are also unknown except some interfaces. 

To this end, we propose a novel framework for **training efficient deep neural networks by exploiting generative adversarial networks** (GANs) . To be specific, the pre-trained teacher networks are regarded as a fixed discriminator and the generator is utilized for derivating training samples which can obtain the maximum response on the discriminator. Then, an efficient network with smaller model size and computational complexity is trained using the generated data and the teacher network, simultaneously. 

Efficient student networks learned using the proposed Data-Free Learning (DFL) method achieve 92.22% and 74.47% accuracies without any training data on the CIFAR-10 and CIFAR-100 datasets, respectively. Meanwhile, our student network obtains an 80.56% accuracy on the CelebA benchmark.

![1582972208798](D:\Notes\raw_images\1582972208798.png)

1. 只使用于分类任务， 实验均为小型数据集分类实验。
2. 网络结构可完全改变。
3. 学生网络仍需要测试集进行效果验证。

对于当前的网络模型压缩和加速算法来说，它们都需要大量的训练数据。然后很多情况下训练数据并不可用，而且当前的模型压缩算法的性能比原始网络低。为此，作者提出了一种利用生成对抗网络(GANs)训练高效深度神经网络的新框架。首先，预先训练好的教师网络作为固定的判别器，并利用该判别器生成训练样本。然后，利用生成的数据和教师网络，训练出尺寸较小、计算复杂度较低的高效网络。

**Teacher Student Interactions**

知识蒸馏（Knowledge Distillation）是当前应用较为广泛的方法，一般是把复杂网络的输出迁移到一个更简单的网络。这篇文章作者使用了三部分损失函数达到Data-Free Learning目的。**Stage 1：**输入一组随机向量，使用生成器G生成图像，然后通过教师网络优化生成器。

生成器——教师网络的损失函数：Loss（1）： ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7Boh%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_i+%5Cmathbb%7BH%7D_%7Bcross%7D%28y_T%5Ei%2Ct_i%29) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BH%7D_%7Bcross%7D%28%C2%B7%29) 是交叉熵损失， ![[公式]](https://www.zhihu.com/equation?tex=y_S%5Ei) 和 ![[公式]](https://www.zhihu.com/equation?tex=+y_T%5Ei) 分别表示学生网络和教师网络的输出， ![[公式]](https://www.zhihu.com/equation?tex=t_i%3Darg%5Cmax_j%28y_T%5Ei%29_j) 。如果生成器G生成的图像与教师网络的训练数据分布相同，那么它们的输出也应该与训练数据具有相似的输出。因此使用one-hot loss促使教师网络生成的图像输出接近one-hot like vectors。也就是说，期望生成与教师网络完全兼容的合成图像，而不是适用于任何场景的一般真实图像。

Loss（2）:  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7B%5Calpha%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5C%7Cf_T%5Ei%5C%7C_1) ， ![[公式]](https://www.zhihu.com/equation?tex=f_T%5Ei) 表示教师网络全连接层前的特征输出。教师网络中的滤波器用来提取训练数据中的固有模式，因此如果输入真实图像，而不是一些随机的向量，特征图往往会收到更高的激活。

Loss（3）： ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7Bie%7D%3D-%5Cmathbb%7BH%7D_%7Binfo%7D%28%5Cfrac%7B1%7D%7Bn%7D%5Csum_iy_T%5Ei%29) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BH%7D_%7Binfo%7D%28p%29%3D-%5Cfrac%7B1%7D%7Bk%7D%5Csum_ip_ilog%28p_i%29) 表示所拥有的信息量，所有变量都为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bk%7D) 时得到最大值。因此，最小化 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7Bie%7D) 能够得到一组类别数量均衡的生成样本。

最终教师网络的损失函数： ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7Btotal%7D%3D%5Cmathbb%7BL%7D_%7Boh%7D%2B%5Calpha+%5Cmathbb%7BL%7D_%7B%5Calpha%7D%2B%5Cbeta+%5Cmathbb%7BL%7D_%7Bie%7D) 。

**Stage 2：**输入一组随机向量，使用生成器G生成图像，得到教师网络和学生网络的输出，计算知识蒸馏损失：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BL%7D_%7BKD%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_i+%5Cmathbb%7BH%7D_%7Bcross%7D%28y_S%5Ei%2Cy_T%5Ei%29) 

![img](https://pic1.zhimg.com/v2-a1636d5d6fa264173e8b5ae1e8a0e030_b.jpg)



## PU Positive Unlabeled Compression

NeurIPS 2019 paper [Positive-Unlabeled Compression on the Cloud](https://arxiv.org/pdf/1909.09757.pdf).

[![img](https://github.com/huawei-noah/Data-Efficient-Model-Compression/raw/master/pu_compress/figure/1.PNG)](https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/pu_compress/figure/1.PNG)

PU Compression is a compression method with little training data. More details can be found at [pu_compress](https://github.com/huawei-noah/DAFL/tree/master/pu_compress).

![1583150823342](D:\Notes\raw_images\1583150823342.png)

