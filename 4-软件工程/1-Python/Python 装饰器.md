## 装饰器

==装饰器本质是一个Python函数，它可以让其它函数在没有任何代码变动的情况下增加额外功能==。有了装饰器，我们可以抽离出大量和函数功能本身无关的雷同代码并继续重用。经常用于具有切面需求的场景：包括插入日志、性能测试、事务处理、缓存和权限校验等。

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

### 编写简单装饰器

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



### 装饰器的调用顺序

装饰器可以叠加使用，==若多个装饰器同时装饰一个函数，那么装饰器的调用顺序和@语法的声明顺序相反==，也就是先进后出规则：

当多个装饰器装饰一个函数时，执行时的顺序是：最先装饰的装饰器，最后一个执行，即==从下往上加载，从上往下调用==。

```python
@decorator1
@decorator2
def func():
    pass

# 等效于：
func = decorator1(decorator2(func()))
```



**实例 1：**

```python
# 通过对运行结果的分析，我们可以发现，当多个装饰器装饰一个函数时，
# 执行时的顺序是：最先装饰的装饰器，最后一个执行。它遵循了先进后出规则 类似于stack
def set_fun1(func):
    print("set_fun1已被定义")  # 打印用于验证在多个装饰器的情况下，多个装饰器之间的执行顺序

    def call_fun1(*args, **kwargs):
        print("call_fun1执行了")  # 当被装饰函数执行时，会打印
        return func()

    return call_fun1


def set_fun2(func):
    print("set_fun2已被定义")

    def call_fun2(*args, **kwargs):
        print("call_fun2执行了")
        return func()

    return call_fun2


# 装饰函数
@set_fun2
@set_fun1
def test():
    print("******如果你看出这是样样老师的装饰器样本，恭喜你找到组织了******")
    
"""
输出：
set_fun1已被定义
set_fun2已被定义
call_fun2执行了
call_fun1执行了
******如果你看出这是样样老师的装饰器样本，恭喜你找到组织了******
"""
```



**实例 2：**

```python
def makebold(fn):
    print("set_fun2已被定义")
    def wrapped():
        print("call_fun2执行了")
        return "<b>" + fn() + "<b>"
    return wrapped

def makeitalic(fn):
    print("set_fun1已被定义")
    def wrapped():
        print("call_fun1执行了")
        return "<i>" + fn() + "<i>"
    return wrapped

@makebold
@makeitalic
def hello():
    return "hello world"
print(hello())

"""
输出：
set_fun1已被定义
set_fun2已被定义
call_fun2执行了
call_fun1执行了
<b><i>hello world<i><b>

Process finished with exit code 0

"""
```



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

Python中，常见的类装饰器包括：@staticmathod、@classmethod 和 @property

- @staticmethod：类的静态方法，跟成员方法的区别是没有self参数，并且可以在类不进行实例化的情况下调用。
- @classmethod：==跟成员方法的区别是接收的第一个参数不是self，而是 cls==（当前类的具体类型）
- @property：表示可以==直接通过类实例直接访问的信息==。

以上，是本次整理的Python高级用法，本文将持续更新。

