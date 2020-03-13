## Python高级用法总结

文章来源：https://www.cnblogs.com/ybjourney/p/8463058.html

> Python很棒，它有很多高级用法值得细细思索，学习使用。本文将根据日常使用，总结介绍Python的一组高级特性，包括：列表推导式、迭代器和生成器、装饰器。

## 列表推导（list comprehensions）

> 场景1：将一个三维列表中所有一维数据为a的元素合并，组成新的二维列表。

最简单的方法：新建列表，遍历原三维列表，判断一维数据是否为a，若为a,则将该元素append至新列表中。
缺点：代码太繁琐，对于Python而言，执行速度会变慢很多。
针对场景1，我们首先应该想到用**列表解析式**来解决处理，一行代码即可解决：

```
lista = [item for item in array if item[0] == 'a']
```

那么，何为列表解析式？
官方解释：列表解析式是Python内置的非常**简单**却**强大**的可以用来创建list的生成式。
**强大具体如何体现？**
可以看到，使用列表解析式的写法更加简短，除此之外，因为是Python内置的用法，底层使用C语言实现，相较于编写Python代码而言，运行速度更快。

> 场景2: 对于一个列表，既要遍历索引又要遍历元素。

这里可以使用Python内建函数enumerate，在循环中更好的获取获得索引。

```
array = ['I', 'love', 'Python']
for i, element in enumerate(array):
    array[i] = '%d: %s' % (i, seq[i])
```

可以使用列表推导式对其进行重构：

```
def getitem(index, element):
    return '%d: %s' % (index, element)

array = ['I', 'love', 'Python']
arrayIndex = [getitem(index, element) for index, element in enumerate(array)]
```

据说这种写法更加的Pythonic。

> 总结：如果要对现有的可迭代对象做一些处理，然后生成新的列表，使用列表推导式将是最便捷的方法。

## 迭代器和生成器

### 迭代器（Iterator）

这里的迭代可以指for循环，在Python中，对于像list，dict和文件等而言，都可以使用for循环，但是它们并不是迭代器，它们属于可迭代对象。
**什么可迭代对象**
最简单的解释：可以使用for...in...语句进行循环的对象，就是可迭代对象（Iterable），可以使用isinstance()方法进行判断。

```
from collections import Iterable 
type = isinstance('python', Iterable)
print type
```

**什么是迭代器**
迭代器指的是可以使用next()方法来回调的对象，可以对可迭代对象使用iter()方法，将其转换为迭代器。

```
temp = iter([1, 2, 3])
print type(temp)
print next(temp)
```

此时temp就是一个迭代器。所以说，迭代器基于两个方法：

- next：返回下一个项目
- *iter* 返回迭代器本身

可理解为可被next()函数调用并不断返回下一个值的对象就是迭代器，在定义一个装饰器时将需要同时定义这两个方法。

**迭代器的优势**

在构建迭代器时，不是将所有的元素一次性的加载，而是等调用next方法时返回元素，所以不需要考虑内存的问题。
**迭代器应用场景**

那么，具体在什么场景下可以使用迭代器呢？

- 数列的数据规模巨大
- 数列有规律，但是不能使用列表推导式描述。

### 生成器

生成器是一种高级迭代器，使得需要返回一系列元素的函数所需的代码更加的简单和高效（不像创建迭代器代码那般冗长）。
**生成器函数**
生成器函数基于yield指令，可以暂停一个函数并返回中间结果。当需要一个将返回一个序列或在循环中执行的函数时，就可以使用生成器，因为当这些元素被传递到另一个函数中进行后续处理时，一次返回一个元素可以有效的提升整体性能。
常见的应用场景是使用生成器的流数据缓冲区。

**生成器表达式**
生成式表达式是一种实现生成器的便捷方式，将列表推导式的中括号替换为圆括号。
和列表推导式的区别：列表生成式可以直接创建一个表，但是生成器表达式是一种边循环边计算，使得列表的元素可以在循环过程中一个个的推算出来，不需要创建完整的列表，从而节省了大量的空间。

```
g = (x * x for x in range(10))
```

> 总结：生成器是一种高级迭代器。生成器的优点是延迟计算，一次返回一个结果，这样非常适用于大数据量的计算。但是，使用生成器必须要注意的一点是：生成器只能遍历一次。

## lambda表达式（匿名函数）

lambda表达式纯粹是为了编写简单函数而设计，起到了一个函数速写的作用，使得简单函数可以更加简洁的表示。
**lambda和def的区别**
lambda表达式可以省去定义函数的过程，让代码更加的简洁，适用于简单函数，编写处理更大业务的函数需要使用def定义。
**lambda表达式常搭配map(), reduce(), filter()函数使用**

- map(): map函数接受两个参数，一个是函数，一个是序列，其中，函数可以接收一个或者多个参数。map将传入的函数依次作用于序列中的每个元素，将结果作为新的列表返回。

```
#将一个列表中的数字转换为字符串
map(str, [1,2,3,4,5,6])
```

- reduce()：函数接收两个参数，一个是函数，另一个是序列，但是，函数必须接收两个参数reduce把结果继续和序列的下一个元素做累积计算，其效果就是reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)。
- filter()：该函数用于筛选，将传入的函数，依次作用于每个元素，然后根据函数的返回值是True还是False，决定是留下还是丢弃该元素。

## 装饰器

装饰器本质是一个Python函数，它可以让其它函数在没有任何代码变动的情况下增加额外功能。有了装饰器，我们可以抽离出大量和函数功能本身无关的雷同代码并继续重用。经常用于具有切面需求的场景：包括插入日志、性能测试、事务处理、缓存和权限校验等。
那么为什么要引入装饰器呢？

> 场景：计算一个函数的执行时间。

一种方法就是定义一个函数，用来专门计算函数的运行时间，然后运行时间计算完成之后再处理真正的业务代码，代码如下：

```
import time 

def get_time(func):
    startTime = time.time()
    func()
    endTime = time.time()
    processTime = (endTime - startTime) * 1000
    print "The function timing is %f ms" %processTime

def myfunc():
    print "start func"
    time.sleep(0.8)
    print "end func"

get_time(myfunc)
myfunc()
```

但是这段代码的逻辑破坏了原有的代码逻辑，就是对所有func函数的调用都需要使用get_time(func)来实现。
那么，有没有更好的展示方式呢？当然有，那就是装饰器。
**编写简单装饰器**
结合上述实例，编写装饰器：

```
def get_time(func):
    def wrapper():
        startTime = time.time()
        func()
        endTime = time.time()
        processTime = (endTime - startTime) * 1000
        print "The function timing is %f ms" %processTime
    return wrapper
    
print "myfunc is:", myfunc.__name__
myfunc = get_time(myfunc)
print "myfunc is: ", myfunc.__name__
myfunc()
```

这样，一个简单的完整的装饰器就实现了，可以看到，装饰器并没有影响函数的执行逻辑和调用。
在Python中，可以使用"@"语法糖来精简装饰器的代码，将上例更改为：

```
@ get_time
def myfunc():
    print "start func"
    time.sleep(0.8)
    print "end func"

print "myfunc is: ", myfunc.__name__
myfunc()
```

** 装饰器的调用顺序**
装饰器可以叠加使用，若多个装饰器同时装饰一个函数，那么装饰器的调用顺序和@语法糖的声明顺序相反，也就是：

```
@decorator1
@decorator2
def func():
    pass
```

等效于：

> func = decorator1(decorator2(func()))

**被装饰的函数带参数**
上述实例中，myfunc()是没有参数的，那如果添加参数的话，装饰器该如何编写呢？

```
#被装饰的函数带参数
def get_time3(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        processTime = (endTime - startTime) * 1000
        print "The function timing is %f ms" %processTime
    return wrapper
@ get_time3
def myfunc2(a):
    print "start func"
    print a
    time.sleep(0.8)
    print "end func"

a = "test"
myfunc2(a)
```

**带参数的装饰器**
装饰器有很大的灵活性，它本身支持参数，例如在上述实例中，@get_time装饰器唯一的参数就是执行业务的函数，当然也可以在装饰器中添加参数，加以逻辑判断。

### 内置装饰器

Python中，常见的类装饰器包括：@staticmathod、@classmethod和@property

- @staticmethod：类的静态方法，跟成员方法的区别是没有self参数，并且可以在类不进行实例化的情况下调用。
- @classmethod：跟成员方法的区别是接收的第一个参数不是self，而是cls（当前类的具体类型）
- @property：表示可以直接通过类实例直接访问的信息。

以上，是本次整理的Python高级用法，本文将持续更新。