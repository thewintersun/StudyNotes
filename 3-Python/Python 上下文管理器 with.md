# 深刻理解上下文管理器

上下文管理器(context manager)是 Python 编程中的重要概念，并且在 Python 3 中得以增强。要加深对 Python 的理解，我们必须要熟练掌握它。然而现在网上的文章少有对其深刻剖析的，建议大家多看 Python 官网文档进行理解。以下是我对 Python 3 文档的解读。

要说上下文管理器，我觉得还是先从 with 语句开始好。（等下说原因）

我们经常会用到 try ... catch ... finally 语句确保一些系统资源得以正确释放。如:

```python
try:
    f = open('somefile')
    for line in f:
        print(line)
except Exception as e:
    print(e)
finally:
    f.close()
```

我们经常用到上面的代码模式，用复用代码的模式来讲，并不够好。于是 with 语句出现了，通过定义一个上下文管理器来封装这个代码块:

```python
with open('somefile') as f:
    for line in f:
        print(line)
```

显然，with 语句比 try 语句好太多了。

## 多上下文管理器

实际上，我们可以同时处理多个上下文管理器：

```python
with A() as a, B() as b:
    suite
```

所以我们大可不必写嵌套 with 语句。

## 上下文管理器类

由于 with 语句利用了上下文管理器，在深入理解 with 语句之前，我们先看看上下文管理器。我们要定义一个上下文管理器其实很简单，==只要一个类实现了\_\_enter\_\_(self)和\_\_exit\_\_(self, exc_type, exc_valye, traceback)就可以了==。

**\_\_enter\_\_(self)** 返回一个对象，可以是当前类的实例，也可以是其他对象。

```python
class SomeThing:
    def __enter__(self):
        return self
    # ...
```

**请注意，我们通常会返回类实例，但这真不是必须的。比如我们有下面的类：**

```python
class LineLength:
    def __init__(self, filepath):
        self.__file = open(self.__filepath)

    def print_line(self):
        for line in self.__file:
            print(len(line), line)

    def __enter__(self):
        return self.__file

    def __exit__(self, exc_type, exc_value, traceback):
        self.__file.close()
        return True
```

我们并没有在\_\_enter\_\_() 中返回 LineLength的实例。实际上，Python 也是这么做的：

```python
In [1]: f = open('/etc/hosts')
In [2]: f
Out[2]: <_io.TextIOWrapper name='/etc/hosts' mode='r' encoding='UTF-8'>
In [3]: f = open('/etc/hosts', 'br')
In [4]: f
Out[4]: <_io.BufferedReader name='/etc/hosts'>
```

## 执行过程

下面让我们看看 with 语句具体是如何执行的。

第一步：执行上下文表达式以获得上下文管理器对象。**上下文表达式**就是 with 和 as 之间的代码。

第二步：加载上下文管理器对象的\_\_exit__()方法，备用。

第三步：执行上下文管理器对象的\_\_enter\_\_()方法。

第四步：将\_\_enter__()方法返回值绑定到 as 后面的 变量中。

第五步：执行 with 内的代码块。

第六步：执行上下文管理器的\_\_exit__()方法。

如果在代码块中发生了异常，异常被传入__exit__()中。如果没有，__exit__()的三个参数会传入 None, None, None。__exit__()需要明确地返回 True 或 False。并且不能在__exit__()中再次抛出被传入的异常，这是解释器的工作，解释器会根据返回值来确定是否继续向上层代码传递异常。当返回 True 时，异常不会被向上抛出，当返回 False 时曾会向上抛出。当没有异常发生传入__exit__()时，解释器会忽略返回值。**问题思考：如果在__exit__()中发生异常呢？**

## contextlib模块

contextlib 模块提供了几个类和函数可以便于我们的常规代码。

**AbstractContextManager：** 此类在 Python3.6中新增，提供了默认的\_\_enter\_\_()和\_\_exit\_\_()实现。\_\_enter\_\_()返回自身，__exit__()返回 None。

**ContextDecorator： 我们可以实现一个上下文管理器，同时可以用作装饰器。**

```python3
class AContext(ContextDecorator):

    def __enter__(self):
        print('Starting')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Finishing')
        return False

# 在 with 中使用
with AContext():
    print('祖国伟大')

# 用作装饰器
@AContext()
def print_sth(sth):
    print(sth)

print_sth('祖国伟大')

#在这两种写法中，有没有发现，第二种写法更好，因为我们减少了一次代码缩进，可读性更强
```

**还有一种好处：当我们已经实现了某个上下文管理器时，只要增加一个继承类，该上下文管理器立刻编程装饰器。**

```python
from contextlib import ContextDecorator
class mycontext(ContextBaseClass, ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
```

**contextmanager：** 我们要实现上下文管理器，总是要写一个类。此函数则容许我们通过一个装饰一个生成器函数得到一个上下文管理器。

```python
import time
from contextlib import contextmanager

@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    time.sleep(3)
    print("</%s>" % name)

>>> with tag("h1"):
...    print("foo")
...
<h1>
foo
</h1>
```

yield 只能返回一次，返回的对象 被绑定到 as 后的变量，不需要返回时可以直接 yield，不带返回值。退出时则从 yield 之后执行。由于contextmanager继承自ContextDecorator，所以被contextmanager装饰过的生成器也可以用作装饰器。

**closing：** 当某对象拥有 close()方法，但不是上下文管理器对象时，为了避免 try 语句，我们可以这样写:

```python
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen('http://www.baidu.com')) as page:
    for line in page:
        print(line)
```

**suppress：**当希望阻止一些异常抛出时，我们可以用：

```python
from contextlib import suppress

with suppress(ImportError):
    import hahaha

# 不好的写法
try:
    import hahaha
except ImportError:
    pass
```

**redirect_stdout、redirect_stderr：**将标准输出、标准错误输出到其他地方

```python
import io

f = io.StringIO()
with redirect_stdout(f):
    help(pow)
s = f.getvalue()
```

为了加深理解，请思考以下问题：

1. 有了 with 语句，我们还要用 try 吗？
2. __enter__()或__exit__()抛出异常的话会发生什么？