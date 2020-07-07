## 总结《Python高级编程》中的知识点

文章来源：https://zhuanlan.zhihu.com/p/33669611



**设置全局变量**

有时候设置全局变量的需求并不是直接赋值，而是想从某个数据结构里引用生成，可以用下面这两种方法，推荐第二种，`golbals()`支持字典用法很方便。

```python
>>> d = {'a': 1, 'b':2}
>>> # 粗暴的写法
>>> for k, v in d.items():
...     exec "{}={}".format(k, v)
...
>>> # 文艺的写法
>>> globals().update(d)
>>> a, b
(1, 2)
>>> 'a', 'b'
('a', 'b')
>>> globals()['a'] = 'b'
>>> a
'b'
```



**字符串格式化**

用`format` 方法可以支持很多种格式，这里就不多说了，可以看官方文档。

```python
>>> "{key}={value}".format(key="a", value=10) # 使⽤命名参数
'a=10'
>>> "[{0:<10}], [{0:^10}], [{0:*>10}]".format("a") # 左中右对⻬
'[a         ], [    a     ], [*********a]'
>>> "{0.platform}".format(sys) # 成员
'darwin'
>>> "{0[a]}".format(dict(a=10, b=20)) # 字典
'10'
>>> "{0[5]}".format(range(10)) # 列表
'5'
>>> "My name is {0} :-{{}}".format('Fred') # 真得想显示{},需要双{}
'My name is Fred :-{}'
>>> "{0!r:20}".format("Hello")
"'Hello'             "
>>> "{0!s:20}".format("Hello")
'Hello               '
>>> "Today is: {0:%a %b %d %H:%M:%S %Y}".format(datetime.now())
'Today is: Mon Mar 31 23:59:34 2014'
```



**列表去重**

这里讲了两种方法，正常情况下`set`是更好的选择；

（注：这里董大视频讲解有误，方法一单位是1.1微妙，是慢于956纳秒，我也自己测试了，确实两种情况都不如`set`）

```python
>>> test_list = [1, 2, 2, 3, 3, 3]
>>> list({}.fromkeys(test_list).keys())    # 利用构建字典键去重
[1, 2, 3] # 列表去重(test_list)
>>> list(set(test_list))    # 常见set函数去重
[1, 2, 3]

In [2]: %timeit list(set(test_list))
1000000 loops, best of 3: 956 ns per loop
In [3]: %timeit list({}.fromkeys(test_list).keys())
1000000 loops, best of 3: 1.1 µs per loop
In [4]: test_list = [random.randint(1, 50) for i in range(10000)]
In [5]: %timeit list(set(test_list))
1000 loops, best of 3: 271 µs per loop
In [6]: %timeit {}.fromkeys(test_list).keys()
1000 loops, best of 3: 310 µs per loop 
```



**操作字典**

字典是 Python 很常用的数据结构，各种函数和方法要掌握。

```python
>>> dict((["a", 1], ["b", 2])) # ⽤两个序列类型构造字典
{'a': 1, 'b': 2}
>>> dict(zip("ab", range(2)))
{'a': 0, 'b': 1}
>>> dict(map(None, "abc", range(2)))
{'a': 0, 'c': None, 'b': 1}
>>> dict.fromkeys("abc", 1) # ⽤序列做 key,并提供默认 value
{'a': 1, 'c': 1, 'b': 1}
>>> {k:v for k, v in zip("abc", range(3))} # 字典解析
{'a': 0, 'c': 2, 'b': 1}
>>> d = {"a":1, "b":2}
>>> d.setdefault("a", 100) # key 存在,直接返回 value 1 
>>> d.setdefault("c", 200) # key 不存在,先设置,后返回 200 
>>> d
{'a': 1, 'c': 200, 'b': 2}
```



**对字典进行逻辑操作**

只能先转成键值对列表再进行操作，然后转回去；

(注：这里原文是 Python 2 中 viewitems 方法，已经被 items 替代)

```python
>>> d1 = dict(a = 1, b = 2)
>>> d2 = dict(b = 2, c = 3)
>>> d1 & d2    # 字典不⽀支持该操作
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for &: 'dict' and 'dict'
>>> v1 = d1.items()
>>> v2 = d2.items()
>>> dict(v1 & v2)     # 交集
{'b': 2}
>>> dict(v1 | v2)     # 并集
{'a': 1, 'b': 2, 'c': 3}
>>> dict(v1 - v2)     #差集(仅v1有,v2没有的)
{'a': 1}
>>> dict(v1 ^ v2)    # 对称差集 (不会同时出现在 v1 和 v2 中)
{'a': 1, 'c': 3}
>>> ('a', 1) in v1 #判断
True
```



**vars**

返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值，类似 locals()。

```python
>>> vars() is locals()
True
>>> vars(sys) is sys.__dict__    # 可用于找类属性
True
```



**实现上下文管理类**

可以用来自动关闭 DB 连接

```python
>>> import pymongo
>>> class Operation(object):
...     def __init__(self, database,
...                  host='localhost', port=27017):
...         self._db = pymongo.MongoClient(
...                       host, port)[database]
...     def __enter__(self):
...         return self._db
...     def __exit__(self, exc_type, exc_val, exc_tb):
...         self._db.connection.disconnect()
...
>>> with Operation(database='test') as db:
...     print db.test.find_one()
```



**contextlib**

这个模块主要包含一个装饰器`contextmanager` ，作用是可以省去像上面那样改写魔术魔法，也能实现同样的类

```python
>>> @contextlib.contextmanager
... def operation(database, host='localhost', 
                  port=27017):
...     db = pymongo.MongoClient(host, port)[database]
...     yield db
...     db.connection.disconnect()
... 
>>> import pymongo
>>> with operation('test') as db:
...     print(db.test.find_one())
```



**包的构建**

如果包里有一些模块不想被`import *` 这样引用，可以用 __all__ 把允许被引用的放进去；

```python
__all__ = ["add", "x"]
```



某些时候,包内的文件太多,需要分类存放到多个目录中,但⼜不想拆分成新的包或子包。这么做是允许的, 只要在 __init__.py 中⽤ __path__ 指定所有子目录的全路径即可 (子目录可放在包外)，下面这段代码可以自动指定子目录。

```python
from os.path import abspath, join
subdirs = lambda *dirs: [abspath(
    join(__path__[0], sub)) for sub in dirs]
__path__ = subdirs("a", "b")
```



**__slots__**

限制给类实例绑定属性，大量属性时减少内存占用

```python
>>> class User(object):
...     __slots__ = ("name", "age")
...     def __init__(self, name, age):
...         self.name = name
...         self.age = age
...
>>> u = User("Dong", 28)
>>> hasattr(u, "__dict__")
False
>>> u.title = "xxx"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'User' object has no attribute 'title'
```



**@cached_property**

主要实现的功能是，被装饰的类实例方法在第一次调用后，会把值缓存下来，下次再调用会直接从 __dict__ 取结果，避免了多次计算；你可以参考下面的代码实现这个装饰器。

```python
class cached_property(object):
     # from werkzeug.utils import cached_property
     def __init__(self, func, name=None, doc=None):
         self.__name__ = name or func.__name__
         self.__module__ = func.__module__
         self.__doc__ = doc or func.__doc__
         self.func = func
     def __get__(self, obj, type=None):
         if obj is None:
             return self
         value = obj.__dict__.get(self.__name__, _missing)
         if value is _missing:
             value = self.func(obj)
             obj.__dict__[self.__name__] = value
         return value
```



**元类里提前定义类方法**

这样可以像定义基类一样，提前给类定义一些方法。

```python3
>>> class HelloMeta(type):
...     def __new__(cls, name, bases, attrs):
...         def __init__(cls, func):
...             cls.func = func
...         def hello(cls):
...             print 'hello world'
...         t = type.__new__(cls, name, bases, attrs)
...         t.__init__ = __init__
...         t.hello = hello
...         return t     # 最后要返回创建的类
...
>>> class Hello(object):
...     __metaclass__ = HelloMeta
...
>>> h = Hello(lambda x: x+1)
>>> h.hello()
hello world
```



**开发陷阱(一)：可变的默认参数**

把临时变量作为默认参数里确实是不错的办法，但要警惕默认参数必须指向一个不可变类型，否则会踩到下面的坑

```python
>>> def append_to(element, to=[]):
...     to.append(element)
...     return to
... 
>>> my_list = append_to(12)
>>> my_list
[12]
>>> my_other_list = append_to(42)
>>> my_other_list
[12, 42]    # 由于[]是可变类型，上次调用的元素还在里面
# 正确的做法是如下
>>> def append_to(element, to=None):
...     if to is None:
...         to = []
...     to.append(element)
...     return to
```



**开发陷阱(二)：闭包变量绑定**

看懂这个坑，需要先理解闭包，推荐一篇文章；

https://zhuanlan.zhihu.com/p/26934085

下面我更换了 PPT 里的代码，坑看得更清楚一些。

```python
>>> def create():
...     a = []
...     for i in range(4):
...         def demo(x):
...             return x*i
...         a.append(demo)
...     return a
... 
>>> for demo in create():
...     print demo(2)
... # 以为是 0 2 4 6 ，实际却是：
6
6
6
6
```

我查了其他资料找到了原因：这是因为 i 是在闭包的作用域（demo 函数的外层作用域），而 Python 的闭包是迟绑定 ，这意味着闭包中用到的变量的值，是在内部函数被调用时查询得到的；

也就是说，`create()`生成实例时，内部的 for 循环开始，使 i 变量的最终变成了3，当随后循环调用闭包`demo(2)`时，在内部调用的 i 实际都是3，要解决这个问题，可以如下：

```python
>>> def create():
...     a = []
...     for i in range(4):
...         def demo(x, i=i):  # 把i绑定成demo的参数
...             return x*i
...         a.append(demo)
...     return a
... 
>>> # 或者这样:
>>> from functools import partial
>>> from operator import mul
>>> def create_multipliers():
...     return [partial(mul, i) for i in range(5)]
... 
>>> # 另外我发现也可以改成生成器表达式：
>>> def create_multipliers():
...     return (lambda x : i * x for i in range(4))
```