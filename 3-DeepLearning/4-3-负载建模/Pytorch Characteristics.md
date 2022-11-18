## PyTorch 架构无关相关参数

#### 模型级别（Static Analyze)

- Flops count  (可计算)
- Parameters count (可计算)
- Operator type, Operator count (可计算) -> ratio 更为合适
  - 对标 Instruction Mix 
- Branch type (ADD SUB MUL CONCAT SPLIT), Branch Count (可计算)  
- Avg Input num / Avg Output num (可计算) 

> Concat * Input branch num
>
> 跨Cell链接是否有影响：Inception Concat 7分支， 残差结构，DenseNet ?
>
> 个人认为没有影响，都是需要进行内存重新搬运。

- locality = sum(input tensor size， output tensor size) （局部性，可计算）

> This metric reflects both temporal locality and spatial locality in an ideal memory system where a cache hit will occur if the same location was accessed before. Lower values of this metric indicate better locality for the operator。
>
> 作者认为：Operator 的耗时，和Operator的类型差别不大，但和输入的数据大小非常相关。

- Parallelism =  scalar arithmetic operations / total  operations  (并行性，可计算)

> Higher values of this metric express greater available parallelism for the operator. 

#### 模型级别（Simulation)

- Inference Time (可计算)

- Operator calls

- Operator gpu time avg


![1605607252952](D:\Notes\raw_images\1605607252952.png)



**BenchIP**

Memory 

- MemAcc： the number of total memory accesses 
- ReDist ==reuse distance== （不可计算）Node Reuse 
- InMem memory size of input
- OutMem memory size of output
- WghMem memory size of weight

Computation 

- Ops the number of operations   (workset size, problem size, 可计算)
- OpMem the ratio of operations to memory access
- ComPtt computation patterns （RD\Elementwise\EL, D 上可分类为：Cube\Vector\Scalar ）

Control 

- PR branch prediction ratio （不可计算）
- MPR misprediction ratio（不可计算）



 内存并行, 指令并行（MLP\ ILP )   = > barrier  同步指令， 编译之后。

> First, convolution and matrix multiplication operators are similar to each other, and most of them have good locality.
>
> Second, all element-wise operators have identical parallelism while the computation intensity on
> each tensor element can vary significantly. 


