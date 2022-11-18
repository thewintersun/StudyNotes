### GRU与LSTM

参考Understanding LSTM Networks：  http://colah.github.io/posts/2015-08-Understanding-LSTMs/

**一、LSTM（长短期记忆网络）**

LSTM是一种特殊的RNN类型，一般的RNN结构如下图所示，是一种将以往学习的结果应用到当前学习的模型，但是这种一般的RNN存在着许多的弊端。举个例子，如果我们要预测“the clouds are in the sky”的最后一个单词，因为只在这一个句子的语境中进行预测，那么将很容易地预测出是这个单词是sky。在这样的场景中，相关的信息和预测的词位置之间的间隔是非常小的，RNN 可以学会使用先前的信息。   

![1565868642887](D:\Notes\raw_images\1565868642887.png)

```python
rnn = RNN()
y = rnn.step(x) # x is an input vector, y is the RNN's output vector

class RNN:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```

PyTorch implementation:

```python
@torch.jit.script
  def RNN(h, x, W_h, U_h, W_y, b_h, b_y):
    y = []
    for t in range(x.size(0)):
      h = torch.tanh(x[t] @ W_h + h @ U_h + b_h)
      y += [torch.tanh(h @ W_y + b_y)]
      if t % 10 == 0:
        print("stats: ", h.mean(), h.var())
    return torch.stack(y), h
```

标准的RNN结构中只有一个神经元，一个tanh层进行重复的学习，这样会存在一些弊端。例如，在比较长的环境中，例如在“I grew up in France… I speak fluent French”中去预测最后的French，那么模型会推荐一种语言的名字，但是预测具体是哪一种语言时就需要用到很远以前的Franch，这就说明在长环境中相关的信息和预测的词之间的间隔可以是非常长的。在理论上，RNN 绝对可以处理这样的长环境问题。人们可以仔细挑选参数来解决这类问题中的最初级形式，但在实践中，RNN 并不能够成功学习到这些知识。然而，LSTM模型就可以解决这一问题。 

![1565868805388](D:\Notes\raw_images\1565868805388.png)

```python
    def LSTM_Cell(input_val):
        batch_num = input_val.shape[1]

        caches = []
        states = []
        states.append([np.zeros([batch_num, HIDDEN]), np.zeros([batch_num, HIDDEN])])

        for x in input_val:
            c_prev, h_prev = states[-1]

            x = np.column_stack([x, h_prev])  # Here input x and hidden state should stack.
            hf = sigmoid(np.dot(x, wf) + bf)
            hi = sigmoid(np.dot(x, wi) + bi)
            ho = sigmoid(np.dot(x, wo) + bo)
            hc = tanh(np.dot(x, wc) + bc)

            c = hf * c_prev + hi * hc
            h = ho * tanh(c)

            states.append([c, h])
            caches.append([x, hf, hi, ho, hc])

        return caches, states
```

如图所示，标准LSTM模型是一种特殊的RNN类型，在每一个重复的模块中有四个特殊的结构，以一种特殊的方式进行交互。在图中，每一条黑线传输着一整个向量，粉色的圈代表一种pointwise 操作(将定义域上的每一点的函数值分别进行运算)，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。  LSTM模型的核心思想是“细胞状态”。“细胞状态”类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。 

![1565868850727](D:\Notes\raw_images\1565868850727.png)

 LSTM 有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。   

![1565868863414](D:\Notes\raw_images\1565868863414.png)

 Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”。LSTM 拥有三个门，来保护和控制细胞状态。 

![1565868881121](D:\Notes\raw_images\1565868881121.png)

 在LSTM模型中，第一步是决定我们从“细胞”中丢弃什么信息，这个操作由一个忘记门层来完成。该层读取当前输入x和前神经元信息h，由ft来决定丢弃的信息。输出结果1表示“完全保留”，0 表示“完全舍弃”。 

![1565868899890](D:\Notes\raw_images\1565868899890.png)

 第二步是确定细胞状态所存放的新信息，这一步由两层组成。sigmoid层作为“输入门层”，决定我们将要更新的值i；tanh层来创建一个新的候选值向量~Ct加入到状态中。在语言模型的例子中，我们希望增加新的主语到细胞状态中，来替代旧的需要忘记的主语。 

![1565868916766](D:\Notes\raw_images\1565868916766.png)

 第三步就是更新旧细胞的状态，将Ct-1更新为Ct。我们把旧状态与 ft相乘，丢弃掉我们确定需要丢弃的信息。接着加上 it * ~Ct。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。在语言模型的例子中，这就是我们实际根据前面确定的目标，丢弃旧代词的信息并添加新的信息的地方。 

![1565868934163](D:\Notes\raw_images\1565868934163.png)

 最后一步就是确定输出了，这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。在语言模型的例子中，因为语境中有一个代词，可能需要输出与之相关的信息。例如，输出判断是一个动词，那么我们需要根据代词是单数还是负数，进行动词的词形变化。

## 二、GRU（ Gated Recurrent Unit，LSTM变体）

![1565868957994](D:\Notes\raw_images\1565868957994.png)

GRU作为LSTM的一种变体，将忘记门和输入门合成了一个单一的更新门。同样还混合了细胞状态和隐藏状态，加诸其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。

## 三、对比

![1565868973184](D:\Notes\raw_images\1565868973184.png)

**LSTM** 

![1565869030718](D:\Notes\raw_images\1565869030718.png)

**GRN** 

![1565869016767](D:\Notes\raw_images\1565869016767.png)

**LSTM** 

![1565869053832](D:\Notes\raw_images\1565869053832.png)

**GRN**

![1565869069756](D:\Notes\raw_images\1565869069756.png)

**性能对比**

![1565869086029](D:\Notes\raw_images\1565869086029.png)

概括的来说，LSTM和GRU都能通过各种Gate将重要特征保留，保证其在long-term 传播的时候也不会被丢失。 

可以看出，标准LSTM和GRU的差别并不大，但是都比tanh要明显好很多，所以在选择标准LSTM或者GRU的时候还要看具体的任务是什么。 

使用LSTM的原因之一是解决RNN Deep Network的Gradient错误累积太多，以至于Gradient归零或者成为无穷大，所以无法继续进行优化的问题。GRU的构造更简单：比LSTM少一个gate，这样就少几个矩阵乘法。在训练数据很大的情况下GRU能节省很多时间。

文章来源：https://blog.csdn.net/lreaderl/article/details/78022724 