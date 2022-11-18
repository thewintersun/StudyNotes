## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

论文地址：https://arxiv.org/abs/1810.04805

作者：Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

机构： Google AI Language

项目地址：https://github.com/google-research/bert



### 摘要

We introduce a new language representation model called BERT, which stands for ==Bidirectional Encoder Representations from Transformers==. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from ==unlabeled text== by ==jointly conditioning on both left and right context in all layers==. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven （11个）natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

### 介绍

<img src="D:\Notes\raw_images\image-20220830153839024.png" alt="image-20220830153839024" style="zoom:80%;" />

Figure 3: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-toleft LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.

<img src="D:\Notes\raw_images\image-20220830150214754.png" alt="image-20220830150214754" style="zoom:80%;" />

Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.



### 实验

<img src="D:\Notes\raw_images\image-20220830153355225.png" alt="image-20220830153355225" style="zoom:67%;" />

Table 5: Ablation over the pre-training tasks using the BERTBASE architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + NoNSP” model during fine-tuning.

<img src="D:\Notes\raw_images\image-20220830153148596.png" alt="image-20220830153148596" style="zoom:67%;" />



文章链接：https://www.zhihu.com/question/298203515/answer/516170825

BERT的“里程碑”意义在于：证明了==一个非常深的模型可以显著提高NLP任务的准确率，而这个模型可以从无标记数据集中预训练得到==。

既然NLP的很多任务都存在数据少的问题，那么要从无标注数据中挖潜就变得非常必要。在NLP中，一个最直接的有效利用无标注数据的任务就是语言模型，因此很多任务都使用了语言模型作为预训练任务。但是这些模型依然比较“浅”，比如上一个大杀器，AllenNLP的[ELMO](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.05365)也就是三层的BiLSTM。

那么有没有可以胜任NLP任务的深层模型？有，就是Transformer。这两年，Transformer已经在机器翻译任务上取得了很大的成功，并且可以做的非常深。自然地，我们可以用Transformer在语言模型上做预训练。因为Transformer是encoder-decoder结构，语言模型就只需要decoder部分就够了。OpenAI的[GPT](https://link.zhihu.com/?target=https%3A//s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)就是这样。但[decoder](https://www.zhihu.com/search?q=decoder&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A516170825})部分其实并不好。因为我们需要的是一个完整句子的encoder，而decoder的部分见到的都是不完整的句子。所以就有了BERT，利用Transformer的encoder来进行预训练。但这个就比较“反直觉”，一般人想不到了。

**2. 我们再来看下BERT有哪些“[反直觉](https://www.zhihu.com/search?q=反直觉&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A516170825})”的设置？**

ELMO的设置其实是最符合直觉的预训练套路，两个方向的语言模型刚好可以用来预训练一个BiLSTM，非常容易理解。但是受限于LSTM的能力，无法变深了。那如何用Transformer在无标注数据行来做一个预训练模型呢？一个最容易想到的方式就是GPT的方式，事实证明效果也不错。那还有没有“更好”的方式？直观上是没有了。而BERT就用了两个反直觉的手段来找到了一个方法。

(1) ==用比语言模型更简单的任务来做预训练==。直觉上，要做更深的模型，需要设置一个比语言模型更难的任务，而BERT则选择了两个看起来更简单的任务：完形填空和句对预测。

(2) 完形填空任务在直观上很难作为其它任务的预训练任务。在完形填空任务中，需要mask掉一些词，这样预训练出来的模型是有缺陷的，因为在其它任务中不能mask掉这些词。而BERT通过随机的方式来解决了这个缺陷：80%加Mask，10%用其它词随机替换，10%保留原词。这样模型就具备了迁移能力。