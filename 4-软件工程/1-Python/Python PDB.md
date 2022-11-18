# python调试：pdb基本用法

https://www.jianshu.com/p/fb5f791fcb18

https://zhuanlan.zhihu.com/p/37294138

### 使用pdb进行调试：

pdb 是 python 自带的一个包，为 python 程序提供了一种交互的源代码调试功能，主要特性包括设置断点、单步调试、进入函数调试、查看当前代码、查看栈片段、动态改变变量的值等。pdb 提供了一些常用的调试命令，详情见表 1。

- 表 1. pdb 常用命令

| 命令                | 解释                       |
| ------------------- | -------------------------- |
| break 或 b 设置断点 | 设置断点                   |
| continue 或 c       | 继续执行程序               |
| list 或 l           | 查看当前行的代码段         |
| step 或 s           | 进入函数                   |
| return 或 r         | 执行代码直到从当前函数返回 |
| exit 或 q           | 中止并退出                 |
| next 或 n           | 执行下一行                 |
| pp                  | 打印变量的值               |
| help                | 帮助                       |

下面结合具体的实例讲述如何使用 pdb 进行调试。

#### 清单 1. 测试代码示例

```python
 import pdb 
 a = "aaa"
 pdb.set_trace() 
 b = "bbb"
 c = "ccc"
 final = a + b + c 
 print final
```

开始调试：直接运行脚本，会停留在 pdb.set_trace() 处，选择 n+enter 可以执行当前的 statement。在第一次按下了 n+enter 之后可以直接按 enter 表示重复执行上一条 debug 命令。

#### 清单 2. 利用 pdb 调试

```python
[root@rcc-pok-idg-2255 ~]#  python epdb1.py 
 > /root/epdb1.py(4)?() 
 -> b = "bbb"
 (Pdb) n 
 > /root/epdb1.py(5)?() 
 -> c = "ccc"
 (Pdb) 
 > /root/epdb1.py(6)?() 
 -> final = a + b + c 
 (Pdb) list 
  1     import pdb 
  2     a = "aaa"
  3     pdb.set_trace() 
  4     b = "bbb"
  5     c = "ccc"
  6  -> final = a + b + c 
  7     print final 
 [EOF] 
 (Pdb) 
 [EOF] 
 (Pdb) n 
 > /root/epdb1.py(7)?() 
 -> print final 
 (Pdb)
```

退出 debug：使用 quit 或者 q 可以退出当前的 debug，但是 quit 会以一种非常粗鲁的方式退出程序，其结果是直接 crash。

#### 清单 3. 退出 debug

```python
[root@rcc-pok-idg-2255 ~]#  python epdb1.py 
 > /root/epdb1.py(4)?() 
 -> b = "bbb"
 (Pdb) n 
 > /root/epdb1.py(5)?() 
 -> c = "ccc"
 (Pdb) q 
 Traceback (most recent call last): 
  File "epdb1.py", line 5, in ? 
    c = "ccc"
  File "epdb1.py", line 5, in ? 
    c = "ccc"
  File "/usr/lib64/python2.4/bdb.py", line 48, in trace_dispatch 
    return self.dispatch_line(frame) 
  File "/usr/lib64/python2.4/bdb.py", line 67, in dispatch_line 
    if self.quitting: raise BdbQuit 
 bdb.BdbQuit
```

打印变量的值：如果需要在调试过程中打印变量的值，可以直接使用 p 加上变量名，但是需要注意的是打印仅仅在当前的 statement 已经被执行了之后才能看到具体的值，否则会报 NameError: < exceptions.NameError … ....> 错误。

#### 清单 4. debug 过程中打印变量

```python
[root@rcc-pok-idg-2255 ~]#  python epdb1.py 
 > /root/epdb1.py(4)?() 
 -> b = "bbb"
 (Pdb) n 
 > /root/epdb1.py(5)?() 
 -> c = "ccc"
 (Pdb) p b 
'bbb'
 (Pdb) 
'bbb'
 (Pdb) n 
 > /root/epdb1.py(6)?() 
 -> final = a + b + c 
 (Pdb) p c 
'ccc'
 (Pdb) p final 
 *** NameError: <exceptions.NameError instance at 0x1551b710 > 
 (Pdb) n 
 > /root/epdb1.py(7)?() 
 -> print final 
 (Pdb) p final 
'aaabbbccc'
 (Pdb)
```

使用 c 可以停止当前的 debug 使程序继续执行。如果在下面的程序中继续有 set_statement() 的申明，则又会重新进入到 debug 的状态，读者可以在代码 print final 之前再加上 set_trace() 验证。

#### 清单 5. 停止 debug 继续执行程序

```python
[root@rcc-pok-idg-2255 ~]#  python epdb1.py 
 > /root/epdb1.py(4)?() 
 -> b = "bbb"
 (Pdb) n 
 > /root/epdb1.py(5)?() 
 -> c = "ccc"
 (Pdb) c 
 aaabbbccc
```

显示代码：在 debug 的时候不一定能记住当前的代码块，如要要查看具体的代码块，则可以通过使用 list 或者 l 命令显示。list 会用箭头 -> 指向当前 debug 的语句。

#### 清单 6. debug 过程中显示代码

```python
[root@rcc-pok-idg-2255 ~]#  python epdb1.py 
 > /root/epdb1.py(4)?() 
 -> b = "bbb"
 (Pdb) list 
  1     import pdb 
  2     a = "aaa"
  3     pdb.set_trace() 
  4  -> b = "bbb"
  5     c = "ccc"
  6     final = a + b + c 
  7     pdb.set_trace() 
  8     print final 
 [EOF] 
 (Pdb) c 
 > /root/epdb1.py(8)?() 
 -> print final 
 (Pdb) list 
  3     pdb.set_trace() 
  4     b = "bbb"
  5     c = "ccc"
  6     final = a + b + c 
  7     pdb.set_trace() 
  8  -> print final 
 [EOF] 
 (Pdb)
```

在使用函数的情况下进行 debug

#### 清单 7. 使用函数的例子

```python
import pdb 
 def combine(s1,s2):      # define subroutine combine, which... 
    s3 = s1 + s2 + s1    # sandwiches s2 between copies of s1, ... 
    s3 = '"' + s3 +'"'   # encloses it in double quotes,... 
    return s3            # and returns it. 
 a = "aaa"
 pdb.set_trace() 
 b = "bbb"
 c = "ccc"
 final = combine(a,b) 
 print final
```

如果直接使用 n 进行 debug 则到 final=combine(a,b) 这句的时候会将其当做普通的赋值语句处理，进入到 print final。如果想要对函数进行 debug 如何处理呢 ? 可以直接使用 s 进入函数块。函数里面的单步调试与上面的介绍类似。如果不想在函数里单步调试可以在断点处直接按 r 退出到调用的地方。

#### 对函数进行 debug

```python
[root@rcc-pok-idg-2255 ~]# python epdb2.py 
 > /root/epdb2.py(10)?() 
 -> b = "bbb"
 (Pdb) n 
 > /root/epdb2.py(11)?() 
 -> c = "ccc"
 (Pdb) n 
 > /root/epdb2.py(12)?() 
 -> final = combine(a,b) 
 (Pdb) s 
 --Call-- 
 > /root/epdb2.py(3)combine() 
 -> def combine(s1,s2):      # define subroutine combine, which... 
 (Pdb) n 
 > /root/epdb2.py(4)combine() 
 -> s3 = s1 + s2 + s1    # sandwiches s2 between copies of s1, ... 
 (Pdb) list 
  1     import pdb 
  2 
  3     def combine(s1,s2):      # define subroutine combine, which... 
  4  ->     s3 = s1 + s2 + s1    # sandwiches s2 between copies of s1, ... 
  5         s3 = '"' + s3 +'"'   # encloses it in double quotes,... 
  6         return s3            # and returns it. 
  7 
  8     a = "aaa"
  9     pdb.set_trace() 
 10     b = "bbb"
 11     c = "ccc"
 (Pdb) n 
 > /root/epdb2.py(5)combine() 
 -> s3 = '"' + s3 +'"'   # encloses it in double quotes,... 
 (Pdb) n 
 > /root/epdb2.py(6)combine() 
 -> return s3            # and returns it. 
 (Pdb) n 
 --Return-- 
 > /root/epdb2.py(6)combine()->'"aaabbbaaa"'
 -> return s3            # and returns it. 
 (Pdb) n 
 > /root/epdb2.py(13)?() 
 -> print final 
 (Pdb)
```

在调试的时候动态改变值 。在调试的时候可以动态改变变量的值，具体如下实例。需要注意的是下面有个错误，原因是 b 已经被赋值了，如果想重新改变 b 的赋值，则应该使用！ B。

#### 清单 9. 在调试的时候动态改变值

```python
[root@rcc-pok-idg-2255 ~]# python epdb2.py 
 > /root/epdb2.py(10)?() 
 -> b = "bbb"
 (Pdb) var = "1234"
 (Pdb) b = "avfe"
 *** The specified object '= "avfe"' is not a function 
 or was not found along sys.path. 
 (Pdb) !b="afdfd"
 (Pdb)
```

