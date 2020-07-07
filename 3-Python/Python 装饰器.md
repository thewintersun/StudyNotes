## 装饰器

装饰器本质是一个Python函数，它可以让其它函数在没有任何代码变动的情况下增加额外功能。有了装饰器，我们可以抽离出大量和函数功能本身无关的雷同代码并继续重用。经常用于具有切面需求的场景：包括插入日志、性能测试、事务处理、缓存和权限校验等。

那么为什么要引入装饰器呢？

> 场景：计算一个函数的执行时间。

一种方法就是定义一个函数，用来专门计算函数的运行时间，然后运行时间计算完成之后再处理真正的业务代码，代码如下：

```python
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

```python
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

```python
@ get_time
def myfunc():
    print "start func"
    time.sleep(0.8)
    print "end func"

print "myfunc is: ", myfunc.__name__
myfunc()
```



**装饰器的调用顺序**
装饰器可以叠加使用，==若多个装饰器同时装饰一个函数，那么装饰器的调用顺序和@语法糖的声明顺序相反==，也就是：

```python
@decorator1
@decorator2
def func():
    pass
```

等效于：

> func = decorator1(decorator2(func()))



**被装饰的函数带参数**
上述实例中，myfunc()是没有参数的，那如果添加参数的话，装饰器该如何编写呢？

```python
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