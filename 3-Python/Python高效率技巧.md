## 这些 Python 高效率技巧，一般人都不会！

你估计已经看了不少关于 Python 技巧的文章，里面可能会提到变量拆包（unpacking）、局部函数等，但是 Python 还有很多不为人知的高效用法，等待着被人发现。本文将介绍作者纵观全网之后，都属于很少没提及的技巧。

## 清理字符串输入

清理用户输入的问题，几乎适用于我们可能编写的每个程序。通常将字符转换为小写或大写就足够了，这时只需要使用正则即可，但是对于复杂的情况，有一种更好的方法：

```python
user_input = "This\nstring has\tsome whitespaces...\r\n"
character_map = {    
    ord('\n') : ' ',
    ord('\t') : ' ',
    ord('\r') : None
}
user_input.translate(character_map)  # This string has some whitespaces... "
```

在上述示例中，可以看到空格符“ \ n”和“ \ t”已被单个空格替换，而“ \ r”已被完全删除。这是一个简单的示例，但是我们可以更进一步，使用 `unicodedata`包及其 `combining()`函数生成范围更广的映射表，从字符串中删除所有重音符号。

## 迭代器切片

如果您尝试获取迭代器的切片，系统会报 TypeError，提示生成器对象不可下标，但是解决方案很简单：

```python
import itertools
s = itertools.islice(range(50), 10, 20)  # <itertools.islice object at 0x7f70fab88138>
for val in s:
...
```

使用 `itertools.islice`，我们可以创建一个 `islice`对象，该对象是产生所需元素的迭代器。不过，请务必注意，这会消耗所有生成器项，直到切片开始为止，而且还会消耗我们的“ islice”对象中的所有项。

Using `itertools.islice` we can create a `islice` object which is an iterator that produces desired items. It's important to note though, that this consumes all generator items up until the start of slice and also all the items in our `islice` object.

## 跳过可迭代对象的开始

有时候需要处理的文件里，明确存在一些不需要的数据行，但是我们不确定数量，比如说代码中的注释。这时， `itertools` 再次为我们提供了简洁的方案：

```python
string_from_file = """
// Author: ...
// License: ...
//
// Date: ...Actual content...
"""
import itertools
for line in itertools.dropwhile(lambda line: line.startswith("//"), string_from_file.split("\n")):
	print(line)
```

这段代码仅在初始注释部分之后，才会产生数据行。如果我们只想在迭代器的开头丢弃数据，而又不知道有具体数量时，这个方法很有用。

## 仅带关键字参数（kwargs）的函数

有时候，使用仅支持关键字参数的函数可以让代码更加清晰易懂：

```python
def test(*, a, b):
    pass
test("value for a", "value for b")  
# TypeError: test() takes 0 positional arguments...
test(a="value", b="value 2")  # Works...
```

只需要在关键字参数前面再加一个 `*` 参数，就可以轻松实现了。当然，如果还希望再加上位置参数，可以在 `*` 参数前面再增加。

## 创建支持 `with`语句的对象

我们都知道如何打开文件或使用 `with`语句获取锁，但是怎样自己可以实现类似的功能呢？一般来说，我们可以使用 `__enter__`和 `__exit__`方法来实现上下文管理器协议：

```python
class Connection:
    def __init__(self):
        ...
        
    def __enter__(self):
    # Initialize connection...
    
    def __exit__(self, type, value, traceback):
    # Close connection...
    
with Connection() as c:
    # __enter__() executes
    ...
    # conn.__exit__() executes
```

上面是最常见的实现方式，但是还有一种更简单的方法：

```python
from contextlib import contextmanager
@contextmanager
def tag(name):
    print(f"<{name}>")
    yield 
    print(f"") 
    
with tag("h1"):
   print("This is Title.")
```

上面的代码段使用 `contextmanager`管理器装饰器实现了内容管理协议。进入“ with”块时，执行“ tag”函数的第一部分（在“ yield”之前），然后执行 `yield`，最后执行其余部分。

## 用 `__slots__`节省内存

如果程序需要创建大量的类实例，我们会发现程序占用了大量内存。这是因为 Python 使用字典来表示类实例的属性，这样的话创建速度很快，但是很耗内存。如果内存是你需要考虑的一个问题，那么可以考虑使用 `__slots__`：

```python
class Person:    
    __slots__ = ["first_name", "last_name", "phone"]
    def __init__(self, first_name, last_name, phone):
        self.first_name = first_name
        self.last_name = last_name
        self.phone = phone
```

当我们定义 `__slots__`属性时，Python会使用固定大小的数组（占用内存少）来存储属性，而不是字典，这大大减少了每个实例所需的内存。不过使用 `__slots__`还有一些缺点：无法声明任何新属性，我们只能使用 `__slots__`中的那些属性。同样，带有 `__slots__`的类不能使用多重继承。

## 限制CPU和内存使用量

如果不是想优化程序内存或CPU使用率，而是想直接将其限制为某个数值，那么Python也有一个可以满足要求的库：

```python
import signal
import resource
import os

# To Limit CPU time
def time_exceeded(signo, frame):
    print("CPU exceeded...")
    raise SystemExit(1)
    
def set_max_runtime(seconds):
    # Install the signal handler and set a resource limit    
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))    
    signal.signal(signal.SIGXCPU, time_exceeded)

# To limit memory usage
def set_max_memory(size):    
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (size, hard))
```

在这里，我们可以设置了最大cpu运行时间以及最大内存使用限制的两个选项。对于cpu限制，我们首先获得该特定资源（ `RLIMIT_CPU`）的软限制和硬限制，然后使用参数指定的秒数和先前获取的硬限制来设置。

最后，我们注册了一个在超过CPU时间后，让系统退出的信号。至于内存，我们再次获取软限制和硬限制，并使用带有大小参数的 `setrlimit`和硬限制完成配置

## 控制导入的内容

某些语言提供了导出成员（变量，方法，接口）的显式机制，例如Golang，它仅导出以大写字母开头的成员。但是在Python中，所有对象都会导出，除非我们使用 `__all__`：

```python
def foo():
	pass
def bar():
	pass
__all__ = ["bar"]
```

上面的代码段中，只会导出 `bar`函数。另外，如果 `__all__`的值为空，那么不会导出任何函数，而且在导入该模块时系统会报 `AttributeError`。

## 实现比较运算符

如果我们要逐一为某个类实现所有的比较运算符，你肯定会觉得很麻烦，因为要实现的方法还不少，有 `__lt__,__le__,__gt__,` 和 `__ge__`。

其实，Python 提供了一种便捷的实现方式，就是通过 `functools.total_ordering`装饰器。

```python
from functools import total_ordering

@total_ordering
class Number:
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        return self.value < other.value
    def __eq__(self, other):
        return self.value == other.value
print(Number(20) > Number(3))
print(Number(1) < Number(5))
print(Number(15) >= Number(15))
print(Number(10) <= Number(2))
```

这是怎么实现的呢？ `total_ordering`可以用来简化实现类排序的过程。我们只需要定义 `__lt__`和 `__eq__`（这是映射剩余操作的最低要求），然后就交给装饰器去完成剩余的工作了。

## 结语

在日常Python编程时，上述特性并非都是必不可少的和有用的，但是其中某些功能可能会不时派上用场，并能简化冗长且令人讨厌的任务。

还要指出的是，所有这些功能都是Python标准库的一部分，而在我看来，其中一些功能似乎不像是应该在标准库中的功能。

因此，每当你决定要用Python实现某些功能时，都请先在标准库中找一找，如果找不到合适的库，那么可能是因为查找的姿势不对。而且即使标准库里没有，有很大的概率已经存在一个第三方库了！