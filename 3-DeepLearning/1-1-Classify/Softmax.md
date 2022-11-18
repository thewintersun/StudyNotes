```python
class SoftMax(NetNode):
    def __init__(self, x):
        ex = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
        super().__init__(ex / np.sum(ex, axis=1, keepdims=True), x)

    def _back(self, x):
        g = self.data * (np.eye(self.data.shape[0]) - self.data)
        x.g += self.g * g
        super()._back()

class LCE(NetNode):
    def __init__(self, p, t):
        super().__init__(
            np.mean(-np.sum(t.data * np.log(p.data), axis=1)),
            p, t
        )

    def _back(self, p, t):
        p.g += self.g * (p.data - t.data) / t.data.shape[0]
        t.g += self.g * -np.log(p.data) / t.data.shape[0]
        super()._back()
```

After further working on this, I figured out that:

1. The homework implementation combines softmax with cross entropy loss as a matter of choice, while my choice of keeping softmax separate as an activation function is also valid.
2. The homework implementation is indeed missing the derivative of softmax for the backprop pass.
3. The gradient of softmax with respect to its inputs is really the partial of each output with respect to each input:

![1564367547938](D:\Notes\raw_images\1564367547938.png)

So for the vector (gradient) form: 

![1564367576307](D:\Notes\raw_images\1564367576307.png)

Which in my vectorized numpy code is simply:

```python
self.data * (1. - self.data)
```

Where `self.data` is the softmax of the input, previously computed from the forward pass.

