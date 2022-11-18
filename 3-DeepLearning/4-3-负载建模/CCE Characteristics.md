## CCE Characteristics

==内存并行度（Memory-Level Parallelism）==

==指令并行度（ Instruction-level parallelism）==

==Memory Reuse  Distance==

- Instruction type, Instruction count -> ratio 更为合适,  instruction time :  理论值，Profiling值
  - Cube Vector Scalar compute instruction： mad， vadd
  - Memory move instruction
  - Controll instruction
- L0\L1\UB allocated momery size (可计算)
- locality = sum(input tensor size， output tensor size) （局部性，可计算）
- wait time： wait ratio 已有，Move time 已有，wait synchronize time ？



### 计算OP

```
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

```
set_flag(PIPE_V, PIPE_M, EVENT_ID0);
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
wait_flag(PIPE_V, PIPE_M, EVENT_ID1);
```

```
pipe_barrier(PIPE_V);
```



### MLP

memory -level parallelism (MLP)是计算机体系结构中的一个术语，指的是同时有挂起的多个内存操作，特别是缓存丢失或转换后备缓冲区translation lookaside buffer (TLB)丢失的能力。

在单个处理器中，MLP可以被认为是指令级并行(ILP)的一种形式。然而, 独立与超标量体系结构常常被混为一谈, 能够在同一时间执行多个指令,例如如英特尔奔腾处理器五方超标量体系结构,能够开始执行五个不同的微指令在一个给定的周期,但它可以处理四个不同的缓存错过了20个不同的负载在任何时候微指令。

有可能有一台机器，它不是超标量，但仍然有高的MLP。

可以证明，没有ILP(不是超标量)的机器，以非流水线的方式一次执行一条指令，但执行硬件预取(不是软件指令级预取)的机器显示的是MLP(由于多个预取未完成)，而不是ILP。这是因为有多个未完成的内存操作，而不是指令。指令常常与操作合并在一起。

此外，由于并行性，多处理器和多线程计算机系统可以说表现出了MLP和ILP，而不是线程内、单进程、ILP和MLP。但是，我们通常将术语MLP和ILP限制为指从非并行单线程代码中提取这种并行性。

