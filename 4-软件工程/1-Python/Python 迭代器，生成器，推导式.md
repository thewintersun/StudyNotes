# Python 迭代器，生成器，推导式

[TOC]

## Python 迭代器

迭代是Python最强大的功能之一，是访问集合元素的一种方式。

迭代器是==一个可以记住遍历的位置的对象==。

迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。==迭代器只能往前不会后退==。

迭代器有两个基本的方法：**iter()** 和 **next()**。

**迭代器的优势**

在构建迭代器时，不是将所有的元素一次性的加载，而是等调用next方法时返回元素，所以==不需要考虑内存的问题==。

**迭代器应用场景**

那么，具体在什么场景下可以使用迭代器呢？

- 数列的数据规模巨大

- 数列有规律，但是不能使用列表推导式描述。

字符串，列表或元组对象都可用于创建迭代器：

**实例(Python 3.0+)**

```python
>>> list=[1,2,3,4]
>>> it = iter(list)   # 创建迭代器对象
>>> print(next(it))  # 输出迭代器的下一个元素
1
>>> print(next(it))
2
>>>
```

迭代器对象可以使用常规for语句进行遍历：

**实例(Python 3.0+)**

```python
#!/usr/bin/python3  
list=[1,2,3,4] 
it = iter(list)   

# 创建迭代器对象 
for x in it:    
	print(x, end=" ")
```

执行以上程序，输出结果如下：

```
1 2 3 4
```

也可以使用 next() 函数：

**实例(Python 3.0+)**

```python
#!/usr/bin/python3  
import sys   

# 引入 sys 模块  
list=[1,2,3,4] 
it = iter(list)    

# 创建迭代器对象  
while True:    
	try:        
		print (next(it))    
	except StopIteration:        
		sys.exit()
```

执行以上程序，输出结果如下：

```python
1
2
3
4
```



**通过 iterable 对象来迭代**

```
for i in range(1000): pass
```

会导致生成一个 1000 个元素的 List，而代码：

```
for i in xrange(1000): pass
```

则不会生成一个 1000 个元素的 List，而是在每次迭代中返回下一个数值，内存空间占用很小。因为 ==xrange 不返回 List==，而是返回一个 iterable 对象。



### 创建一个迭代器

把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__() 。

如果你已经了解的面向对象编程，就知道类都有一个构造函数，Python 的构造函数为 __init__(),  它会在对象初始化的时候执行。

更多内容查阅：[Python3 面向对象](https://www.runoob.com/python3/python3-class.html)

\__iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 

\__next__() 方法并通过 StopIteration 异常标识迭代的完成。



创建一个返回数字的迭代器，初始值为 1，逐步递增 1：

**实例(Python 3.0+)**

```python
class MyNumbers:  
    def __iter__(self):    
        self.a = 1    
        return self   
    
    def __next__(self):    
        x = self.a    
        self.a += 1    
        return x  

myclass = MyNumbers() 
myiter = iter(myclass)  
print(next(myiter)) 
print(next(myiter)) 
print(next(myiter)) 
print(next(myiter)) 
print(next(myiter))
```

执行输出结果为：

```
1
2
3
4
5
```

### StopIteration

==StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况==，在 \__next__() 方法中我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。

在 20 次迭代后停止执行：

**实例(Python 3.0+)**

```python
class MyNumbers:  
    def __iter__(self):    
        self.a = 1    
        return self   
    
    def __next__(self):    
        if self.a <= 20:      
            x = self.a      
            self.a += 1      
            return x    
        else:      
            raise StopIteration  

myclass = MyNumbers() 
myiter = iter(myclass)  
for x in myiter:  
    print(x)

```

执行输出结果为：

```python
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
```



## Python 生成器

在 Python 中，==使用了 yield 的函数被称为生成器==（generator）。

跟普通函数不同的是，==生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器==。

在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值,  并在下一次执行 next() 方法时从当前位置继续运行。

调用一个生成器函数，返回的是一个迭代器对象。



以下实例使用 yield 实现斐波那契数列：

**实例(Python 3.0+)**

```python
#!/usr/bin/python3  
import sys  

def fibonacci(n): 
	# 生成器函数 - 斐波那契    
	a, b, counter = 0, 1, 0    
	while True:        
		if (counter > n):             
			return yield a 
		a, b = b, a + b        
		counter += 1 

f = fibonacci(10)  # f 是一个迭代器，由生成器返回生成  
while True:    
	try:        
		print (next(f), end=" ")    
	except StopIteration:        
		sys.exit()

```

执行以上程序，输出结果如下：

```python
0 1 1 2 3 5 8 13 21 34 55
```



**使用 yield 的第四版**

**实例**

```python
#!/usr/bin/python 
# -*- coding: UTF-8 -*-  
def fab(max):     
    n, a, b = 0, 0, 1     
    while n < max:         
        yield b      
        # 使用 yield        
        # print b         
        a, b = b, a + b         
        n = n + 1  

for n in fab(5):     
    print n
```

第四个版本的 fab 和第一版相比，仅仅把 print b 改为了 yield b，就在保持简洁性的同时获得了 iterable 的效果。

调用第四版的 fab 和第二版的 fab 完全一致：

```python
1 
1 
2 
3 
5
```

简单地讲，==yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable 对象==！在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时，fab 函数就返回一个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。

也可以手动调用 fab(5) 的 next() 方法（因为 fab(5) 是一个 generator 对象，该对象具有 next() 方法），这样我们就可以更清楚地看到 fab 的执行流程：

**清单 6. 执行流程**

```python
>>>f = fab(5)  
>>> f.next()  1  
>>> f.next()  1  
>>> f.next()  2  
>>> f.next()  3  
>>> f.next()  5  
>>> f.next()  
Traceback (most recent call last):  
    File "<stdin>", line 1, in <module>  StopIteration
```

当函数执行结束时，generator 自动抛出 StopIteration 异常，表示迭代完成。在 for 循环里，无需处理 StopIteration 异常，循环会正常结束。

**我们可以得出以下结论：**

==一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行==。虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。

yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。

**实例**

```python
import time


def task1():
    while True:
        print("1-before")
        yield
        print("1-after")
        time.sleep(0.5)


def task2():
    while True:
        print("2-before")
        yield
        print("2-after")
        time.sleep(0.5)


t1 = task1()
t2 = task2()

for _ in range(0, 2):
    next(t1)
    next(t2)

'''
1-before
2-before
1-after
1-before
2-after
2-before
'''
```

```python
def func(x):
    x = yield x
    x = yield x  # x is None


m = func(3)
for x in m:
    print(x, end=' ')

""" 
3 None 
"""
```

### 使用 isgeneratorfunction 判断

```python
>>>from inspect import isgeneratorfunction  
>>> isgeneratorfunction(fab)  
True
```

要注意区分 fab 和 fab(5)，fab 是一个 generator function，而 fab(5) 是调用 fab 返回的一个 generator，好比类的定义和类的实例的区别。

### 类的定义和类的实例

```python
>>>import types  
>>> isinstance(fab, types.GeneratorType)  
False  
>>> isinstance(fab(5), types.GeneratorType)  
True
```

fab 是无法迭代的，而 fab(5) 是可迭代的。

```python
>>>from collections import Iterable  
>>> isinstance(fab, Iterable)  
False  
>>> isinstance(fab(5), Iterable)  
True
```

每次调用 fab 函数都会生成一个新的 generator 实例，各实例互不影响：

```python
>>>f1 = fab(3)  
>>> f2 = fab(5)  
>>> print 'f1:', f1.next()  
f1: 1  
>>> print 'f2:', f2.next()  
f2: 1  
>>> print 'f1:', f1.next()  
f1: 1  
>>> print 'f2:', f2.next()  
f2: 1  
>>> print 'f1:', f1.next()  
f1: 2  
>>> print 'f2:', f2.next()  
f2: 2  
>>> print 'f2:', f2.next()  
f2: 3  
>>> print 'f2:', f2.next()  
f2: 5
```

### return 的作用

在一个 generator function 中，如果没有 return，则默认执行至函数完毕，如果==在执行过程中 return，则直接抛出 StopIteration 终止迭代==。



## Python 推导式

Python 推导式是一种独特的数据处理方式，可以从一个数据序列构建另一个新的数据序列的结构体。

那么，何为列表解析式？
官方解释：列表解析式是Python内置的非常**简单**却**强大**的可以用来创建list的生成式。

**强大具体如何体现？**
可以看到，使用列表解析式的写法更加简短，除此之外，==因为是Python内置的用法，底层使用C语言实现==，相较于编写Python代码而言，运行速度更快。

Python 支持各种数据结构的推导式：

- 列表(list)推导式
- 字典(dict)推导式
- 集合(set)推导式
- 元组(tuple)推导式

### 列表推导式

列表推导式格式为：

```python
[表达式 for 变量 in 列表] 
[out_exp_res for out_exp in input_list]

或者 

[表达式 for 变量 in 列表 if 条件]
[out_exp_res for out_exp in input_list if condition]
```

- out_exp_res：列表生成元素表达式，可以是有返回值的函数。
- for out_exp in input_list：迭代 input_list 将 out_exp 传入到 out_exp_res 表达式中。
- if condition：条件语句，可以过滤列表中不符合条件的值。

过滤掉长度小于或等于3的字符串列表，并将剩下的转换成大写字母：

**实例**

```python
>>> names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
>>> new_names = [name.upper() for name in names if len(name)>3]
>>> print(new_names)
['ALICE', 'JERRY', 'WENDY', 'SMITH']
```

计算 30 以内可以被 3 整除的整数：

**实例**

```python
>>> multiples = [i for i in range(30) if i % 3 == 0]
>>> print(multiples)
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
```



### 字典推导式

字典推导基本格式：

```python
{ key_expr: value_expr for value in collection }

或

{ key_expr: value_expr for value in collection if condition }
```

使用字符串及其长度创建字典：

**实例**

```python
listdemo = ['Google','Runoob', 'Taobao']
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
>>> newdict = {key:len(key) for key in listdemo}
>>> newdict
{'Google': 6, 'Runoob': 6, 'Taobao': 6}
```

提供三个数字，以三个数字为键，三个数字的平方为值来创建字典：

**实例**

```python
>>> dic = {x: x**2 for x in (2, 4, 6)}
>>> dic
{2: 4, 4: 16, 6: 36}
>>> type(dic)
<class 'dict'>
```



### 集合推导式

集合推导式基本格式：

```python
{ expression for item in Sequence }
或
{ expression for item in Sequence if conditional }
```

计算数字 1,2,3 的平方数：

**实例**

```python
>>> setnew = {i**2 for i in (1,2,3)}
>>> setnew
{1, 4, 9}
```

判断不是 abc 的字母并输出：

**实例**

```python
>>> a = {x for x in 'abracadabra' if x not in 'abc'}
>>> a
{'d', 'r'}
>>> type(a)
<class 'set'>
```



### 元组推导式

元组推导式可以利用 range 区间、元组、列表、字典和集合等数据类型，快速生成一个满足指定需求的元组。

元组推导式基本格式：

```python
(expression for item in Sequence )
或
(expression for item in Sequence if conditional )
```

元组推导式和列表推导式的用法也完全相同，只是元组推导式是用 **()** 圆括号将各部分括起来，而列表推导式用的是中括号 **[]**，另外==元组推导式返回的结果是一个生成器对象==。

例如，我们可以使用下面的代码生成一个包含数字 1~9 的元组：

**实例**

```python
>>> a = (x for x in range(1,10))
>>> a
<generator object <genexpr> at 0x7faf6ee20a50> # 返回的是生成器对象

>>> tuple(a)    # 使用 tuple() 函数，可以直接将生成器对象转换成元组
(1, 2, 3, 4, 5, 6, 7, 8, 9)
```

