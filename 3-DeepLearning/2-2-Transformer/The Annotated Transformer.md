## The Annotated Transformer

文章地址：https://zhuanlan.zhihu.com/p/339207092

原文地址： http://nlp.seas.harvard.edu/2018/04/03/attention.html

Attention is All You Need[1] 一文中提出的Transformer网络结构最近引起了很多人的关注。Transformer不仅能够明显地提升翻译质量，还为许多NLP任务提供了新的结构。虽然原文写得很清楚，但实际上大家普遍反映很难正确地实现。

所以我们为此文章写了篇注解文档，并给出了一行行实现的Transformer的代码。本文档删除了原文的一些章节并进行了重新排序，并在整个文章中加入了相应的注解。此外，本文档以Jupyter notebook的形式完成，本身就是直接可以运行的代码实现，总共有400行库代码，在4个GPU上每秒可以处理27,000个tokens。

想要运行此工作，首先需要安装PyTorch[2]。这篇文档完整的notebook文件及依赖可在github[3] 或 Google Colab[4]上找到。

### **0.准备工作**

```python
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")

%matplotlib inline
```

**本文注解部分都是以引用的形式给出的，主要内容都是来自原文。**

### 1. 背景

减少**序列处理任务的计算量**是一个很重要的问题，也是Extended Neural GPU、ByteNet和ConvS2S等网络的动机。上面提到的这些网络都以CNN为基础，并行计算所有输入和输出位置的隐藏表示。

在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数随位置间的距离增长而增长，比如ConvS2S呈线性增长，ByteNet呈现以对数形式增长，这会使学习较远距离的两个位置之间的依赖关系变得更加困难。而在Transformer中，**操作次数则被减少到了常数级别。**

Self-attention有时候也被称为Intra-attention，是在单个句子不同位置上做的Attention，并得到序列的一个表示。它能够很好地应用到很多任务中，包括阅读理解、摘要、文本蕴涵，以及独立于任务的句子表示。端到端的网络一般都是基于循环注意力机制而不是序列对齐循环，并且已经有证据表明在简单语言问答和语言建模任务上表现很好。

据我们所知，==Transformer是第一个完全依靠Self-attention而不使用序列对齐的RNN或卷积的方式来计算输入输出表示的转换模型==。

### 2.模型结构

目前大部分比较热门的神经序列转换模型都有Encoder-Decoder结构[9]。Encoder将输入序列 $(x_1,....,x_n)$映射到一个连续表示序列 $z=(z_1,....z_n)$ 。

对于编码得到的 $z$，Decoder每次解码生成一个符号，直到生成完整的输出序列： $(y_1,....y_m)$ 。对于每一步解码，模型都是自回归的[10]，即在生成下一个符号时将先前生成的符号作为附加输入。

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

Transformer的整体结构如下图所示，在Encoder和Decoder中都使用了Self-attention, Point-wise和全连接层。Encoder和decoder的大致结构分别如下图的左半部分和右半部分所示。

![img](D:\Notes\raw_images\v2-22a369f0f1b0d542ced248dcb215b6e8_720w.jpg)

***Encoder***

Encoder由N=6个相同的层组成。

```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

我们在每两个子层之间都使用了残差连接(Residual Connection) [11]和归一化 [12]。

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

也就是说，每个子层的输出为 $LayerNorm(x+Sublayer(x)) $，其中 $ Sublayer(x) $是由子层自动实现的函数。我们在每个子层的输出上使用Dropout，然后将其添加到下一子层的输入并进行归一化。

为了能方便地使用这些残差连接，模型中所有的子层和Embedding层的输出都设定成了相同的维度，即 $d_{model}=512$。

```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```

每层都有两个子层组成。第一个子层实现了“多头”的 Self-attention，第二个子层则是一个简单的Position-wise的全连接前馈网络。

```python
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

***Decoder***

Decoder也是由N=6个相同层组成。

```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

除了每个编码器层中的两个子层之外，**解码器还插入了第三种子层对编码器栈的输出实行“多头”的Attention。**与编码器类似，我们在每个子层两端使用残差连接进行短路，然后进行层的规范化处理。

```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

我们还修改解码器中的Self-attention子层以防止当前位置Attend到后续位置。这种Masked的Attention是考虑到输出Embedding会偏移一个位置，==确保了生成位置i的预测时，仅依赖小于i的位置处的已知输出，相当于把后面不该看到的信息屏蔽掉==。

```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

> 下面的Attention mask图显示了允许每个目标词（行）查看的位置（列）。在训练期间，当前解码位置的词不能Attend到后续位置的词。

```python
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```

![img](D:\Notes\raw_images\v2-3c315c4be3eac87af1ed64eead1024ca_720w.jpg)

***Attention***

Attention函数可以将Query和一组Key-Value对映射到输出，其中Query、Key、Value和输出都是向量。 输出是值的加权和，其中分配给每个Value的权重由Query与相应Key的兼容函数计算。

我们称这种特殊的Attention机制为"Scaled Dot-Product Attention"。输入包含维度为 $d_k$ 的Query和Key，以及维度为$d_v$ 的Value。 我们首先分别计算Query与各个Key的点积，然后将每个点积除以 $\sqrt {d_k}$，最后使用Softmax函数来获得Key的权重。

![img](D:\Notes\raw_images\v2-e551f16cc7511f55151d152d28e2aab8_720w.jpg)

在具体实现时，我们可以以矩阵的形式进行并行运算，这样能加速运算过程。具体来说，将所有的Query、Key和Value向量分别组合成矩阵 Q、K 和 V，这样输出矩阵可以表示为：
$$
Attention (Q,K,V)=softmax⁡( \frac {QK^T}{\sqrt{d_k}}) V
$$

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

两种最常用的Attention函数是**加和**Attention[13]和**点积**（乘积）Attention，我们的算法与点积Attention很类似，但是$\frac1{d_k}$ 的比例因子不同。加和Attention使用具有单个隐藏层的前馈网络来计算兼容函数。虽然两种方法理论上的复杂度是相似的，但在实践中，点积Attention的运算会更快一些，也更节省空间，因为它可以使用高效的矩阵乘法算法来实现。

虽然对于较小的 $d_k$， 这两种机制的表现相似，但在不放缩较大的 $d_k$ 时，加和Attention要优于点积Attention[14]。我们怀疑，对于较大的 $d_k$，点积大幅增大, 将Softmax函数推向具有极小梯度 的区域（为了阐明点积变大的原因，假设 q和 k是独立的随机变量, 平均值为 0，方差为 1，这样他们的点积为 $ q⋅k= \sum_i^{d_k} q_ik_i$，同样是均值 0为方差为 $d_k$ ）。为了抵消这种影响，我们用 $\frac1{d_k}$ 来缩放点积。

![img](D:\Notes\raw_images\v2-344bd404818d76829bcd5e7c27a9e780_720w.jpg)

“多头”机制能让模型考虑到不同位置的Attention，另外“多头”Attention可以在不同的子空间表示不一样的关联关系，使用单个Head的Attention一般达不到这种效果。
$$
MultiHead (Q,K,V)= Concat ( head_1,…, head_h)W^O \\ where \ head_i = Attention (QW_i^Q,KW_i^K,VW_i^V)
$$
其中参数矩阵为 $W_i^Q∈R^{d_{model} ×d_k}，W_i^K∈R^{d_{model} ×d_k},W_i^V∈R^{d_{model} ×d_v} 和W^O∈R^{hd_v×d_{model}}$ 。

我们的工作中使用 h=8个Head并行的Attention，对每一个Head来说有$ d_k=d_v=d_{model} /h = 512/8 = 64$, 总计算量与完整维度的单个Head的Attention很相近。

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

***4.Attention在模型中的应用***

Transformer中以三种不同的方式使用了“多头”Attention：

1) 在"Encoder-Decoder Attention"层，Query来自先前的解码器层，并且Key和Value来自Encoder的输出。Decoder中的每个位置Attend输入序列中的所有位置，这与Seq2Seq模型中的经典的Encoder-Decoder Attention机制[15]一致。

2) Encoder中的Self-attention层。在Self-attention层中，所有的Key、Value和Query都来同一个地方，这里都是来自Encoder中前一层的输出。Encoder中当前层的每个位置都能Attend到前一层的所有位置。

3) 类似的，解码器中的Self-attention层允许解码器中的每个位置Attend当前解码位置和它前面的所有位置。这里需要屏蔽解码器中向左的信息流以保持自回归属性。具体的实现方式是在缩放后的点积Attention中，屏蔽（设为负无穷）Softmax的输入中所有对应着非法连接的Value。

***Position-wise前馈网络***

除了Attention子层之外，Encoder和Decoder中的每个层都包含一个全连接前馈网络，分别地应用于每个位置。其中包括两个线性变换，然后使用ReLU作为激活函数。
$$
FFN⁡(x)=max(0,xW_1+b_1)W_2+b_2
$$
虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。这其实是相当于使用了两个内核大小为1的卷积。这里设置输入和输出的维数为 $d_{model}=512$，内层的维度为 $d_{ff}=2048$。

```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

***Embedding 和 Softmax***

与其他序列转换模型类似，我们使用预学习的Embedding将输入Token序列和输出Token序列转化为dmodel维向量。我们还使用常用的预训练的线性变换和Softmax函数将解码器输出转换为预测下一个Token的概率。在我们的模型中，我们在两个Embedding层和Pre-softmax线性变换之间共享相同的权重矩阵，类似于[16]。在Embedding层中，我们将这些权重乘以$\sqrt {d_{model}} $。

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

***7.位置编码***

由于我们的模型不包含递归和卷积结构，为了==使模型能够有效利用序列的顺序特征，我们需要加入序列中各个Token间相对位置或Token在序列中绝对位置的信息==。在这里，我们将位置编码添加到编码器和解码器栈底部的输入Embedding。由于位置编码与Embedding具有相同的维度dmodel，因此两者可以直接相加。其实这里还有许多位置编码可供选择，其中包括可更新的和固定不变的[17]。

在此项工作中，我们使用不同频率的正弦和余弦函数：

$$
PE_{(pos,2i)}=sin⁡(pos/10000^{2i/d_{model}} ) \\

PE_{(pos,2i+1)}=cos⁡(pos/10000^{2i/d_{model}} )
$$
其中 pos是位置, $i$ 是维度。也就是说，位置编码的每个维度都对应于一个正弦曲线, 其波长形成从 $2π$到 $10000⋅2π$的等比级数。我们之所以选择了这个函数，是因为我们假设它能让模型很容易学会Attend相对位置, 因为对于任何固定的偏移量 $k$, $PE_{pos+k} $可以表示为 PEpos的线性函数。

此外，在编码器和解码器堆栈中，我们在Embedding与位置编码的加和上都使用了Dropout机制。 在基本模型上, 我们使用 $P_{drop} =0.1$ 的比率。

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```

> 如下所示，位置编码将根据位置添加正弦曲线。曲线的频率和偏移对于每个维度是不同的。

```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
```

![img](D:\Notes\raw_images\v2-cbc029fc2ae39ff329d1d51ef056e491_720w.jpg)

我们也尝试了使用预学习的位置Embedding，但是发现这两个版本的结果基本是一样的。我们选择正弦曲线版本的实现，因为使用此版本能让模型能够处理大于训练语料中最大序了使用列长度的序列。

***完整模型***

> 下面定义了连接完整模型并设置超参的函数。

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# Small example model.
tmp_model = make_model(10, 10, 2)
None
```

### **训练**

本节介绍模型的训练方法。

> 快速穿插介绍训练标准编码器解码器模型需要的一些工具。首先我们定义一个包含源和目标句子的批训练对象用于训练，同时构造掩码。

**批和掩码**

```python
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

> 接下来，我们创建一个通用的训练和得分函数来跟踪损失。我们传入一个通用的损失计算函数，它也处理参数更新。

***训练循环***

```python
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```

***训练数据和批处理***

我们使用标准WMT 2014英语-德语数据集进行了训练，该数据集包含大约450万个句子对。 使用字节对的编码方法对句子进行编码，该编码具有大约37000个词的共享源-目标词汇表。 对于英语-法语，我们使用了WMT 2014 英语-法语数据集，该数据集由36M个句子组成，并将词分成32000个词片(Word-piece)的词汇表。

句子对按照近似的序列长度进行批处理。每个训练批包含一组句子对，包含大约25000个源词和25000个目标词。

> 我们将使用torch text来创建批次。下面更详细地讨论实现过程。 我们在torchtext的一个函数中创建批次，确保填充到最大批训练长度的大小不超过阈值（如果我们有8个GPU，则阈值为25000）。

```python
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

***硬件和训练进度***

我们在一台配备8个NVIDIA P100 GPU的机器上训练我们的模型。 对于使用本文所述的超参数的基本模型，每个训练单步大约需要0.4秒。 我们对基础模型进行了总共100,000步或12小时的训练。 对于我们的大型模型，每个训练单步时间为1.0秒。 ==大型模型通常需要训练300,000步（3.5天）==。

***优化器***

我们选择Adam[1]作为优化器，其参数为 $β_1=0.9,β_2=0.98 \  和 \ ϵ=10^{−9}$. 根据以下公式，我们在训练过程中改变了学习率：

$$
lrate =d_{model}^ {−0.5}⋅min(step\_num^{−0.5},step\_num⋅warmup\_steps^{−1.5})
$$
在预热中随步数线性地增加学习速率，并且此后与步数的反平方根成比例地减小它。我们设置预热步数为4000。

> 注意：这部分非常重要，需要这种设置训练模型。

```python
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

> 当前模型在不同模型大小和超参数的情况下的曲线示例。

```python
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```

![img](D:\Notes\raw_images\v2-93b10b26d465fac8a3358977c738ed56_720w.jpg)

***正则化***

**标签平滑**

在训练期间，我们采用了值 $ϵ_{ls}=0.1 $[2]的标签平滑。 这种做法提高了困惑度，因为模型变得更加不确定，但提高了准确性和BLEU分数。

> 我们使用 $KL \ div \ loss$ 实现标签平滑。 相比使用独热目标分布，我们创建一个分布，其包含正确单词的置信度和整个词汇表中分布的其余平滑项。

```python
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

在这里，我们可以看到标签平滑的示例。

```python
# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
```

![img](D:\Notes\raw_images\v2-05f1e8b6d5fa8ebf49e83919aa19e3a5_720w.jpg)

> 如果对给定的选择非常有信心，标签平滑实际上会开始惩罚模型。

```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
    #print(predict)
    return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```

![img](D:\Notes\raw_images\v2-79853c622ae8110bb253169c8f8cfde5_720w.jpg)

## **第一个例子**

> 我们可以先尝试一个简单的复制任务。 给定来自小词汇表的随机输入符号集，目标是生成那些相同的符号。

***数据生成***

```python
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
```

***损失计算***

```python
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
```

***贪心解码***

```python
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))

Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
1.9274832487106324
Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
1.657595729827881
Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
1.4888023376464843
Epoch Step: 1 Loss: 1.682198 Tokens per Sec: 432.092305
Epoch Step: 1 Loss: 1.313169 Tokens per Sec: 639.441857
1.3485562801361084
Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
0.9853351473808288
Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
0.5686767101287842
Epoch Step: 1 Loss: 0.997076 Tokens per Sec: 433.009746
Epoch Step: 1 Loss: 0.343118 Tokens per Sec: 642.288427
0.34273059368133546
Epoch Step: 1 Loss: 0.459483 Tokens per Sec: 434.594030
Epoch Step: 1 Loss: 0.290385 Tokens per Sec: 642.519464
0.2612409472465515
Epoch Step: 1 Loss: 1.031042 Tokens per Sec: 434.557008
Epoch Step: 1 Loss: 0.437069 Tokens per Sec: 643.630322
0.4323212027549744
Epoch Step: 1 Loss: 0.617165 Tokens per Sec: 436.652626
Epoch Step: 1 Loss: 0.258793 Tokens per Sec: 644.372296
0.27331129014492034
```

> 为简单起见，此代码使用贪心解码来预测翻译。

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    1     2     3     4     5     6     7     8     9    10
[torch.LongTensor of size 1x10]
```

## **真实示例**

> 现在我们通过IWSLT德语-英语翻译任务介绍一个真实示例。 该任务比上文提及的WMT任务小得多，但它说明了整个系统。 我们还展示了如何使用多个GPU处理加速其训练。

```python
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
```

***数据加载***

> 我们将使用torchtext和spacy加载数据集以进行词语切分。

```python
# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```

> 批训练对于速度来说很重要。我们希望批次分割非常均匀并且填充最少。 要做到这一点，我们必须修改torchtext默认的批处理函数。 这部分代码修补其默认批处理函数，以确保我们搜索足够多的句子以构建紧密批处理。

***迭代器***

```python
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler) 
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
```

***多 GPU训练***

> 最后为了真正地快速训练，我们将使用多个GPU。 这部分代码实现了多GPU字生成。 它不是Transformer特有的，所以我不会详细介绍。 其思想是将训练时的单词生成分成块，以便在许多不同的GPU上并行处理。 我们使用PyTorch并行原语来做到这一点：

- 复制 - 将模块拆分到不同的GPU上
- 分散 - 将批次拆分到不同的GPU上
- 并行应用 - 在不同GPU上将模块应用于批处理
- 聚集 - 将分散的数据聚集到一个GPU上
- nn.DataParallel - 一个特殊的模块包装器，在评估之前调用它们。

```python
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```

> 现在我们创建模型，损失函数，优化器，数据迭代器和并行化。

```python
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
```

> 现在我们训练模型。 我将稍微使用预热步骤，但其他一切都使用默认参数。 在具有4个Tesla V100 GPU的AWS p3.8xlarge机器上，每秒运行约27,000个词，批训练大小大小为12,000。

***训练系统***

```python
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt

if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
```

> 一旦训练完成，我们可以解码模型以产生一组翻译。 在这里，我们只需翻译验证集中的第一个句子。 此数据集非常小，因此使用贪婪搜索的翻译相当准确。

```python
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break

Translation: <unk> <unk> . In my language , that means , thank you very much . 
Gold: <unk> <unk> . It means in my language , thank you very much . 
```

***附加组件：BPE，搜索，平均***

> 所以这主要涵盖了Transformer模型本身。 有四个方面我们没有明确涵盖。 我们还实现了所有这些附加功能 OpenNMT-py[3].
> 1) ==字节对编码/ 字片(Word-piece)==：我们可以使用库来首先将数据预处理为子字单元。参见 Rico Sennrich 的 subword-nmt 实现[4]。这些模型将训练数据转换为如下所示：

▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

> 2) ==共享嵌入==：当使用具有共享词汇表的BPE时，我们可以在源/目标/生成器之间共享相同的权重向量，详细见[5]。 要将其添加到模型，只需执行以下操作：

```python
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
```

> 3) ==集束搜索==：这里展开说有点太复杂了。 PyTorch版本的实现可以参考 OpenNMT- py[6]。
> 4) ==模型平均==：这篇文章平均最后k个检查点以创建一个集合效果。 如果我们有一堆模型，我们可以在事后这样做：

```python
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
```

### **结果**

在WMT 2014英语-德语翻译任务中，大型Transformer模型（表2中的Transformer（大））优于先前报告的最佳模型（包括集成的模型）超过2.0 BLEU，建立了一个新的最先进BLEU得分为28.4。 该模型的配置列于表3的底部。在8个P100 GPU的机器上，训练需要需要3.5天。 甚至我们的基础模型也超过了之前发布的所有模型和集成，而且只占培训成本的一小部分。

在WMT 2014英语-法语翻译任务中，我们的大型模型获得了41.0的BLEU分数，优于以前发布的所有单一模型，不到以前最先进技术培训成本的1/4 模型。 使用英语到法语训练的Transformer（大）模型使用dropout概率$P_{drop}=0.1$，而不是0.3。

![img](D:\Notes\raw_images\v2-2e5f88879ea24edde2c3b0354f2bdb41_720w.jpg)

> 我们在这里编写的代码是基本模型的一个版本。 这里有系统完整训练的版本 (Example Models[7]).
> 通过上一节中的附加扩展，OpenNMT-py 复制在EN-DE WMT上达到 26.9。 在这里，我已将这些参数加载到我们的重新实现中。

```python
!wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
model, SRC, TGT = torch.load("en-de-model.pt")
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)

Translation:    <s> ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E \- Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .
```

***注意力可视化***

> 即使使用贪婪的解码器，翻译看起来也不错。 我们可以进一步想象它，看看每一层注意力发生了什么。

```python
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
Encoder Layer 2
```

![img](D:\Notes\raw_images\v2-f3879c7feb25e8745056c055931ed9d5_720w.jpg)

```
Encoder Layer 4
```

![img](D:\Notes\raw_images\v2-43d60cbd3f8be5bc267027d5c3bdf75f_720w.jpg)

```
Encoder Layer 6
```

![img](D:\Notes\raw_images\v2-31e5e8cafd53a656b71b477a9a1f1139_720w.jpg)

```
Decoder Self Layer 2
```

![img](D:\Notes\raw_images\v2-b424f7090bd397da4f0287862d11b7fc_720w.jpg)



```
Decoder Src Layer 2
```

![img](D:\Notes\raw_images\v2-75ac6362fdf6de04a36b7b2cb894d269_720w.jpg)

```
Decoder Self Layer 4
```

![img](D:\Notes\raw_images\v2-dd59f315c4428631cef87a88dbb30794_720w.jpg)

```
Decoder Src Layer 4
```

![img](D:\Notes\raw_images\v2-756a2dd84968de750788d073c2475db3_720w.jpg)

```
Decoder Self Layer 6
```

![img](D:\Notes\raw_images\v2-b076f5da48e2744f82cc51172dfe7574_720w.jpg)

```
Decoder Src Layer 6
```

![img](D:\Notes\raw_images\v2-7dcb607e20974f21c05fd35c98f875db_720w.jpg)

### **结论**

> 希望这段代码对未来的研究很有用。 如果您有任何问题，请与我们联系。 如果您发现此代码有用，请查看我们的其他OpenNMT工具。

```python
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

### **参考链接**

[1] [https://arxiv.org/abs/1412.6980](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1412.6980)

[2] [https://arxiv.org/abs/1512.00567](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.00567)

[3] [https://github.com/opennmt/opennmt-py](https://link.zhihu.com/?target=https%3A//github.com/opennmt/opennmt-py)

[4] [https://github.com/rsennrich/subword-nmt](https://link.zhihu.com/?target=https%3A//github.com/rsennrich/subword-nmt)

[5] [https://arxiv.org/abs/1608.05859](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1608.05859)

[6] [https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py](https://link.zhihu.com/?target=https%3A//github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py)

[7] [http://opennmt.net/Models-py/](https://link.zhihu.com/?target=http%3A//opennmt.net/Models-py/)