## The Unreasonable Effectiveness of Recurrent Neural Networks

文章地址：https://karpathy.github.io/2015/05/21/rnn-effectiveness/

-- Andrej Karpathy blog

### Recurrent Neural Networks
A glaring limitation of Vanilla Neural Networks (and also Convolutional Networks) is that their API is too constrained: they accept a fixed-sized vector as input (e.g. an image) and produce a fixed-sized vector as output (e.g. probabilities of different classes).Not only that: These models perform this mapping using a fixed amount of computational steps (e.g. the number of layers in the model). 
- VNN or CNN 都有明显的一个局限性fixed-sized input and output, 甚至是固定的运算步骤。


**RNN computation.** So how do these things work? At the core, RNNs have a deceptively simple API: They accept an input vector x and give you an output vector y. However, crucially this output vector’s contents are influenced not only by the input you just fed in, but also on the entire history of inputs you’ve fed in in the past. Written as a class, the RNN’s API consists of a single step function:
```
rnn = RNN()
y = rnn.step(x) # x is an input vector, y is the RNN's output vector
```
The RNN class has some internal state that it gets to update every time step is called. In the simplest case this state consists of a single hidden vector h. Here is an implementation of the step function in a Vanilla RNN:
```
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
```
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
The above specifies the forward pass of a vanilla RNN. This RNN’s parameters are the three matrices W_hh, W_xh, W_hy. _The hidden state self.h is initialized with the zero vector_. The np.tanh function implements a non-linearity that squashes the activations to the range [-1, 1]. Notice briefly how this works: There are two terms inside of the tanh: one is based on the previous hidden state and one is based on the current input. In numpy np.dot is matrix multiplication. The two intermediates interact with addition, and then get squashed by the tanh into the new state vector. 

![1565868673906](D:\Notes\raw_images\1565868673906.png)

**Going deep.** For instance, we can form a 2-layer recurrent network as follows:
```
y1 = rnn1.step(x)
y = rnn2.step(y1)
```
In other words we have two separate RNNs: One RNN is receiving the input vectors and the second RNN is receiving the output of the first RNN as its input. 

**Getting fancy.** I’d like to briefly mention that in practice most of us use a slightly different formulation than what I presented above called a Long Short-Term Memory (LSTM) network. The LSTM is a particular type of recurrent network that works slightly better in practice, owing to its more powerful update equation and some appealing backpropagation dynamics. I won’t go into details, but everything I’ve said about RNNs stays exactly the same, *except the mathematical form for computing the update (the line self.h = ... ) gets a little more complicated*. 

### Bidirectional RNN
Bidirectional RNNs are based on the idea that the output at time t may not only depend on the previous elements in the sequence, but also future elements. For example, to predict a missing word in a sequence you want to look at both the left and the right context. Bidirectional RNNs are quite simple. They are just two RNNs stacked on top of each other. The output is then computed based on the hidden state of both RNNs.

![1565868420167](D:\Notes\raw_images\1565868420167.png)

### Deep (Bidirectional) RNNs
Deep (Bidirectional) RNNs are similar to Bidirectional RNNs, only that we now have multiple layers per time step. In practice this gives us a higher learning capacity (but we also need a lot of training data).

![1565868464193](D:\Notes\raw_images\1565868464193.png)
