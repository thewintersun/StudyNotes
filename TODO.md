Face Recognition
论文：GhostVLAD for set-based face recognition
地址：https://arxiv.org/abs/1810.09951
阅读：https://mp.weixin.qq.com/s/R1hdkPTdFCo7JvOKNcEzJg 
--------------------------------------------------------------------------------

Data Clean 
论文：Few-Example Object Detection with Model Communication
地址：https://arxiv.org/pdf/1706.08249.pdf
论文：Towards Human-Machine Cooperation: Self-supervised Sample Mining for Object Detection
地址：http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3145.pdf

0 - The Unreasonable Effectiveness of Recurrent Neural Networks
The fall of RNN / LSTM
https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0

The Unreasonable Effectiveness of Recurrent Neural Networks
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

There’s something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I’ve in fact reached the opposite conclusion). Fast forward about a year: I’m training RNNs all the time and I’ve witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you.
We’ll train RNNs to generate text character by character and ponder the question “how is that even possible?”
By the way, together with this post I am also releasing code on Github that allows you to train character-level language models based on multi-layer LSTMs. You give it a large chunk of text and it will learn to generate text like it one character at a time. You can also use it to reproduce my experiments below. But we’re getting ahead of ourselves; What are RNNs anyway?

Attention and Augmented Recurrent Neural Network
https://distill.pub/2016/augmented-rnns/
https://github.com/kjw0612/awesome-rnn

RNN Blog:
http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 

RNN computation. So how do these things work? At the core, RNNs have a deceptively simple API: They accept an input vector x and give you an output vector y. However, crucially this output vector’s contents are influenced not only by the input you just fed in, but also on the entire history of inputs you’ve fed in in the past. Written as a class, the RNN’s API consists of a single step function:

博客地址： https://gist.github.com/karpathy/d4dee566867f8291f086 


RNN: 其他博客教程
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
https://zhuanlan.zhihu.com/p/22930328
http://www.deeplearning.net/tutorial/rnnslu.html#rnnslu
http://www.deeplearning.net/tutorial/lstm.html#lstm

--------------------------------------------------------------------------------

LSTM介绍：
http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 


--------------------------------------------------------------------------------

World2Vec 介绍：
Google 地址：https://code.google.com/archive/p/word2vec/


论文：Sequence to Sequence Learning with Neural Networks
论文地址：https://arxiv.org/pdf/1409.3215.pdf

An Intuitive Explanation of Connectionist Temporal Classification
CRNN
论文：An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition
论文地址：https://arxiv.org/abs/1507.05717

论文：Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
论文地址：http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.6306&rep=rep1&type=pdf
作者：Alex Graves
机构：Istituto Dalle Molle di Studi sull’Intelligenza Artificiale (IDSIA), Galleria 2, 6928 Manno-Lugano, Switzerland

摘要：
This paper presents a novel method for training RNNs to label unsegmented sequences directly, thereby solving both problems. 

介绍：
RNN 相对于HMM 和 CRF的优点：
Recurrent neural networks (RNNs), on the other hand, require no prior knowledge of the data, beyond thechoice of input and output representation. 
They can be trained discriminatively, and their internal state provides a powerful, general mechanism for modelling time series. 
In addition, they tend to be robust to temporal and spatial noise.
RNN的问题：
The problem is that the standard neural network objective functions are defined separately for each point in the training sequence; in other words, RNNs can only be trained to make a series of independent label classifications. This means that the training data must be pre-segmented, and that the network outputs must be post-processed to give the final label sequence.

This paper presents a novel method for labelling sequence data with RNNs that removes the need for presegmented training data and post-processed outputs, and models all aspects of the sequence within a single network architecture. 
The basic idea is to interpret the network outputs as a probability distribution over all possible label sequences, conditioned on a given input sequence. Given this distribution, an objective function can be derived that directly maximises the probabilities of the correct labellings. Since the objective function is differentiable, the network can then be trained with standard backpropagation through time
(Werbos, 1990).
论文：SEE: Towards Semi-Supervised End-to-End Scene Text Recognition
论文地址：https://arxiv.org/pdf/1712.05404.pdf 
代码：https://github.com/Bartzi/see 

论文：Sequence to Sequence Learning for Optical Character Recognition
论文地址：https://arxiv.org/abs/1511.04176

Detection
作者：Abhinav Shrivastava, Abhinav Gupta, Ross Girshick
Training Region-based Object Detectors with Online Hard Example Mining
论文： https://arxiv.org/abs/1604.03540
论文：Path Aggregation Network for Instance Segmentation
论文地址：https://arxiv.org/abs/1803.01534论文：Path Aggregation Network for Instance Segmentation
论文地址：https://arxiv.org/abs/1803.01534
SSD
论文：
https://arxiv.org/abs/1512.02325

mxnet源码：
https://github.com/zhreshold/mxnet-ssd 
https://github.com/dmlc/mxnet/tree/master/example/ssd

YOLOV3
博客：https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
代码：https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
论文：https://arxiv.org/abs/1804.02767
softmax loss是我们最熟悉的loss之一了，分类任务中使用它，分割任务中依然使用它。softmax loss实际上是由softmax和cross-entropy loss组合而成，两者放一起数值计算更加稳定。这里我们将其数学推导一起回顾一遍。
https://zhuanlan.zhihu.com/p/34044634

令z是softmax层的输入，f(z)是softmax的输出，则

import numpy as np

def softmax(x):
    # We need to subtract the max to avoid numerical issues
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()


单个像素i的softmax loss等于cross-entropy error如下:

展开上式：

在caffe实现中，z即bottom blob，l(y,z)是top blob，反向传播时，就是要根据top blob diff得到bottom blob diff,所以要得到 

下面求loss对z的第k个节点的梯度
https://deepnotes.io/softmax-crossentropy

初始化 - Xavier - MSRA
