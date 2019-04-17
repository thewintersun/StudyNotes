### Paper Resources

Paper: Deep Density-based Image Clustering

Link: https://arxiv.org/abs/1812.04287

Comment: In this paper, we propose a two-stage deep density-based image clustering (DDC) framework to address these issues. The first stage is to train a deep convolutional autoencoder (CAE) to extract low-dimensional feature representations from high-dimensional image data, and then apply t-SNE to further reduce the data to a 2-dimensional space favoring density-based clustering algorithms. The second stage is to apply the developed density-based clustering technique on the 2-dimensional embedded data to automatically recognize an appropriate number of clusters with arbitrary shapes. Concretely, a number of local clusters are generated to capture the local structures of clusters, and then are merged via their density relationship to form the final clustering result. Experiments demonstrate that the proposed DDC achieves comparable or even better clustering performance than state-of-the-art deep clustering methods, even though the number of clusters is not given.

<hr/>
