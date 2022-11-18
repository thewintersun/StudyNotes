## Meta Matrix Factorization for Federated Rating Predictions

论文地址：https://dl.acm.org/doi/10.1145/3397271.3401081

作者：Yujie Lin, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Dongxiao Yu, Jun Ma, Maarten de Rijke, Xiuzhen Cheng

机构：山东大学，University of Amsterdam

发表：SIGIR '20

代码地址：https://github.com/TempSDU/MetaMF

元 矩阵分解 for 联邦 概率预测



### 摘要

凭借优越的隐私保护优势，联邦推荐变得越来越可行。然而，之前关于联邦推荐系统因为他们的模型规模太大，无法在移动设备中轻松运行。此外，现有的联邦推荐需要在每个设备中微调推荐模型，这使得它们难以有效地利用用户/设备之间的协同过滤（CF）信息。

我们设计一个新的联邦学习框架来为这种环境进行Rating Prediction（RP），该框架可与最先进的完全集中式 RP 方法相媲美。为此，我们引入了一种新的联邦矩阵分解 (MF) 框架，称为元矩阵分解 (MetaMF)，它能够使用元网络生成 private item embeddings 和 RP 模型。

给定一个用户，我们首先通过协作记忆 (collaborative memory，CM) 模块收集有用信息来获得协作向量（collaborative vector）。然后，我们使用元推荐器 (meta recommender，MR) 模块来生成private item embeddings和基于服务器中的协作向量的 RP 模型。为了解决生成大量高维 item embeddings 的问题，我们设计了一种升维生成（RG）策略，首先生成一个低维item embeddings矩阵和一个升维矩阵，然后将它们相乘得到高维 embeddings。最后，我们使用生成的模型为她设备中的给定用户生成 private RPs。

我们在四个基准数据集上进行了广泛的实验，以将 MetaMF 与现有的 MF 方法进行比较，发现 MetaMF 可以实现具有竞争力的性能。此外，我们发现 MetaMF 通过更好地利用用户/设备之间的 CF 实现了比现有联邦方法更高的 RP 性能。

![image-20220215164034739](D:\Notes\raw_images\image-20220215164034739.png)

图 1：MetaMF overview。它由三个模块组成。 CM 模块和 具有 RG 策略的 MR 模块, 为每个用户生成 private item embeddings 和 RP 模型，并将其部署到服务器中。预测模块，根据生成的item embeddings和部署到设备中的RP 模型，为每个用户预测private ratings。

