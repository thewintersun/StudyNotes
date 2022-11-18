# LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop



![img](https://www.yf.io/p/lsun/img/teaser_web.jpg)

## Abstract

While there has been remarkable progress in the performance of visual recognition algorithms, the state-of-the-art models tend to be exceptionally data-hungry. Large labeled training datasets, expensive and tedious to produce, are required to optimize millions of parameters in deep network models. Lagging behind the growth in model capacity, the available datasets are quickly becoming outdated in terms of size and density. To circumvent this bottleneck, we propose to amplify human effort through a partially automated labeling scheme, leveraging deep learning with humans in the loop. Starting from a large set of candidate images for each category, we iteratively sample a subset, ask people to label them, classify the others with a trained model, split the set into positives, negatives, and unlabeled based on the classification confidence, and then iterate with the unlabeled set. To assess the effectiveness of this cascading procedure and enable further progress in visual recognition research, we construct a new image dataset, LSUN. It contains around one million labeled images for each of 10 scene categories and 20 object categories. We experiment with training popular convolutional networks and find that they achieve substantial performance gains when trained on this dataset.

## Paper

Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser and Jianxiong Xiao
[LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop](http://arxiv.org/abs/1506.03365)
arXiv:1506.03365 [cs.CV], 10 Jun 2015



数据集官网：

https://www.yf.io/p/lsun



数据集下载：

http://dl.yf.io/lsun/objects/

http://dl.yf.io/lsun/scenes/