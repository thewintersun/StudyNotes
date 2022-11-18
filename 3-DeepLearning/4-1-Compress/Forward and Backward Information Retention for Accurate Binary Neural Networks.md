## Forward and Backward Information Retention for Accurate Binary Neural Networks

论文地址：https://arxiv.org/abs/1909.10788

作者：Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen, Ziran Wei, Fengwei Yu, Jingkuan Song

机构：北航，商汤

发表:  CVPR 2020

总结：通过保留正向和反向梯度中的信息来提升二值化网络的精度。



### 摘要

Weight and activation binarization is an effective approach to deep neural network compression and can accelerate the inference by leveraging bitwise operations. Although many binarization methods have improved the accuracy of the model by minimizing the quantization error in forward propagation, there remains a noticeable performance gap between the binarized model and the full-precision one. 

权值和激活二值化是一种有效的深度神经网络压缩方法，可以利用位运算加快推理速度。虽然许多二值化方法都通过最小化前向传播中的量化误差来提高模型的精度，但二值化模型与全精度模型之间仍存在明显的性能差距。

Our empirical study indicates that the quantization brings information loss in both forward and backward propagation, which is the bottleneck of training accurate binary neural networks. To address these issues, we propose an Information Retention Network (IR-Net) to retain the information that consists in the forward activations and backward gradients. 

我们的实验研究表明，量化在前向和后向传播中都带来了信息损失，这是训练精确二值神经网络的瓶颈。为了解决这些问题，我们提出了一个信息保留网络(IR-Net)来==保留包含在正向激活和反向梯度中的信息==。

IR-Net mainly relies on two technical contributions: (1) Libra Parameter Binarization (Libra-PB): simultaneously minimizing both quantization error and information loss of parameters by balanced and standardized weights in forward propagation; (2) Error Decay Estimator (EDE): minimizing the information loss of gradients by gradually approximating the sign function in backward propagation, jointly considering the updating ability and accurate gradients. 

IR-Net主要依靠两个技术贡献:(1)Libra参数二值化(Libra- pb):在正向传播过程中通过平衡并标准化权值，同时最小化参数的量化误差和信息损失; (2)误差衰减估计(EDE): 在反向传播过程中，通过逐步逼近符号函数（ the sign function）来最小化梯度的信息损失，同时考虑更新能力和梯度的精确性。

We are the first to investigate both forward and backward processes of binary networks from the unified information perspective, which provides new insight into the mechanism of network binarization. Comprehensive experiments with various network structures on CIFAR-10 and ImageNet datasets manifest that the proposed IR-Net can consistently outperform state-of-the-art quantization methods.

首次从统一信息的角度研究了二值化网络的前向和后向过程，为网络二值化机制提供了新的视角。在CIFAR-10和ImageNet数据集上对各种网络结构进行的综合实验表明，所提出的IR-Net始终能够优于最先进的量化方法。

### 实验结果

![1591348248800](D:\Notes\raw_images\1591348248800.png)

![1591348297864](D:\Notes\raw_images\1591348297864.png)