# Python之列表与元组的区别详解

**相同点：都是序列类型**

回答它们的区别之前，先来说说两者有什么相同之处。list 与 tuple 都是序列类型的容器对象，可以存放任何类型的数据、支持切片、迭代等操作

```python
foos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
foos[0:10:2]
[0, 2, 4, 6, 8]

bars = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
bars[1:10:2]
(1, 3, 5, 7, 9)
```

两者的操作如此相似，Python 为什么还要设计一种叫 tuple 的类型出来呢？这就要从它们的不同之处来寻找答案。

**不同点一：不可变 VS 可变**

两种类型除了字面上的区别(括号与方括号)之外，最重要的一点是==tuple是不可变类型，大小固定==，而 list 是可变类型、数据可以动态变化，这种差异使得两者提供的方法、应用场景、性能上都有很大的区别。

列表特有方法：

```python
foo = [2,3,1,9,4]
foo.sort() # 排序
foo.insert(5,10) # 插入
foo.reverse() # 反转
foo.extend([-1, -2]) # 扩展
foo.remove(10) # 移除
foo.pop() # 弹出最后一个元素
foo.append(5) # 追加
```

所有的操作都基于原来列表进行更新，而==tuple 作为一种不可变的数据类型，同样大小的数据，初始化和迭代 tuple 都要快于 list==

```python
python -m timeit “[1,2,3,4,5]”
10000000 loops, best of 3: 0.123 usec per loop

python -m timeit “(1,2,3,4,5)”
100000000 loops, best of 3: 0.0166 usec per loop
```

==同样大小的数据，tuple 占用的内存空间更少==

```python
foo = tuple(range(1000))
bar = list(range(1000))
foo.sizeof()
8024
bar.sizeof()
9088
```

==原子性的 tuple 对象还可作为字典的键==

```python
foo = (1,(2,3))
d = {foo: 1}

bar = (1, [2,3]) # 非原子性tuple，因为元组中包含有不可哈希的list
d = {bar: 1}
Traceback (most recent call last):
File “”, line 1, in
TypeError: unhashable type: ‘list’
```

**不同点二：同构 VS 异构**

==tuple 用于存储异构 (heterogeneous) 数据，当做没有字段名的记录来用==，比如用 tuple 来记录一个人的身高、体重、年龄。

person = (“zhangsan”, 20, 180, 80)
比如记录坐标上的某个点

point = (x, y)
而列表一般用于存储同构数据(homogenous)，同构数据就是具有相同意义的数据，比如下面的都是字符串类型

[“zhangsan”, “Lisi”, “wangwu”]
再比如 list 存放的多条用户记录

[(“zhangsan”, 20, 180, 80), (“wangwu”, 20, 180, 80)]
数据库操作中查询出来的记录就是由元组构成的列表结构。

因为 tuple 作为没有名字的记录来使用在某些场景有一定的局限性，所以又有了一个 namedtuple 类型的存在，namedtuple 可以指定字段名，用来当做一种轻量级的类来使用。
 