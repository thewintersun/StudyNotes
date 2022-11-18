## Python 变量作用域

[TOC]

### 变量作用域

变量作用域（scope）在Python中是一个容易掉坑的地方。

Python的作用域一共有4中，分别是：

•	L（Local）：最内层，包含局部变量，比如一个函数/方法内部。

•	E（Enclosing）：包含了非局部(non-local) 也非全局(non-global) 的变量。比如两个嵌套函数，一个函数（或类） A 里面又包含了一个函数 B ，那么对于 B 中的名称来说 A 中的作用域就为  nonlocal。

•	G（Global）：当前脚本的最外层，比如当前模块的全局变量。

•	B（Built-in）： 包含了内建的变量/关键字等，最后被搜索。

规则顺序： L –> E –> G –> B。即：在局部找不到，便会去局部外的局部找（例如闭包），再找不到就会去全局找，再者去内建中找。

![image-20220419155021479](D:\Notes\raw_images\image-20220419155021479.png)

==Python除了**def/class/lambda** 外，其他如: **if/elif/else/ try/except for/while**并不能改变其作用域==。定义在他们之内的变量，外部还是可以访问。

```python
>>> if True:
...     a = 'I am A'

>>> a

'I am A'
```

定义在if语言中的变量a，外部还是可以访问的。但是需要注意如果 if 被 def/class/lambda 包裹，在内部赋值，就变成了此 **函数/类/lambda** 的局部作用域。

**实例 1**：

```python
g = 1  #全局的
def fun():
    g = 2 #局部的
    return g

print fun()  # 结果为2
print g   # 结果为1
```


但是要注意，有时候想在函数内部引用全局的变量，疏忽了就会出现错误，比如：

```python
var = 1

def fun():
    print(var)

print(fun())
# 1
```

```python
var = 1
def fun():
    print(var)
    var = 200
    
print fun()

"""
Traceback (most recent call last):
  File "D:/Scripts/pythonProject/test.py", line 160, in <module>
    print(fun())
  File "D:/Scripts/pythonProject/test.py", line 156, in fun
    print(var)
UnboundLocalError: local variable 'var' referenced before assignment
"""
```

在未被赋值之前引用的错误！为什么？因为在函数的内部，==解释器探测到var被重新赋值了，所以var成为了局部变量，但是在没有被赋值之前就想使用var，便会出现这个错误==。

```python
var = 1
def fun():
    var = var + 1
    return var

print(fun())

"""
Traceback (most recent call last):
  File "D:/Scripts/pythonProject/test.py", line 157, in <module>
    print(fun())
  File "D:/Scripts/pythonProject/test.py", line 154, in fun
    var = var + 1
UnboundLocalError: local variable 'var' referenced before assignment
"""
```

解决的方法是==在函数内部添加 global var 但运行函数后全局的var也会被修改==。

```python
var = 1

def fun():
    global var
    print(var)
    var = 200

print(fun())
print(var)

"""
1
None
200
"""
```

**实例 2：**

```python
global_var = 1
global_list = []


def sample_func():
    global_var = 2    # global_var使用局部变量
    global_list.append(1)  #  global_list 使用全局变量


if __name__ == '__main__':
    sample_func()
    print(global_var, global_list)

"""
1 [1]
"""
```

**实例3**：

```python
name = "123" 
 
def func2(): 
    name = "456" 
    def func1(): 
        print(name) 
    func1() 
 
func2() 
# 456
```

```python
name = "123" 
 
def func2(): 
    def func1(): 
        print(name) 
    func1() 
    name = "456" 
 
func2()
# 结果：NameError: free variable 'name' referenced before assignment in enclosing scope 赋值在调用之前
```

```python
def outer_fun(): 
    def inner_fun(): 
        a += 1 
        print(a) 
    inner_fun() 
    print(a) 

a = 10 
outer_fun() 
print(a) 
# local variable 'a' referenced before assignment 
```

```python
def outer_fun(): 
    def inner_fun(): 
        print(a) 
    inner_fun() 
    a += 1 
    print(a) 
    
a = 10 
outer_fun() 
print(a) 
# free variable 'a' referenced before assignment in enclosing scope
```

**实例 4**：

```python
x = 12

def f1(a, b=x):
    print(a, b)

x = 15
f1(4)
print(x)

"""
4 12
15
"""
```



### 推导式中的变量的作用域

**实例：**

```python
def handle(x): 
    pass 
     
i = 3 
     
def foo(x): 
    def bar(): 
        return i 
    for i in x: 
        handle(x) 
    return bar() 
  
i = 3 
     
def foo_1(x): 
    def bar(): 
        return i 
    y = [i for i in x] 
    handle(y) 
    return bar() 
     
print('%s, %s' % (foo(['x', 'y']), foo_1(['x', 'y']))) 
     
# y, 3   
# 第一个foo 采用的是局部变量 i
# 第二个应为生成式中的i是生成式函数局部的，所以采用全局变量i的值。

```

**实例**：

在类中不行，在交互解释器和函数里可以

```python
class A:
    a = 10
    b = ['12', 2, 12]
    print(a)                        # 10
    print(b)                        # ['12', 2, 12]
    print([a * element for element in b])     # NameError: name 'a' is not defined

A()
# NameError: name 'a' is not defined
```

```python
class A:
    a = 42
    b = list(a + i for i in range(10))
    print(b)
    
A()
# name 'a' is not defined 
```

```python
def func1():
    a = 42
    b = list(a + i for i in range(10))
    print(b)

func1()
# [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
```



### 函数的变量作用域

变量作用域在函数执行前确定：从上到下，保存**函数体** 至 堆区，**函数名**至栈区指向堆。

**实例：**

```python
def fun1():
    return [lambda x: i * x for i in range(4)]


print(fun1())
print([m(2) for m in fun1()])
"""
[<function fun1.<locals>.<listcomp>.<lambda> at 0x000001B5382792F0>, 
<function fun1.<locals>.<listcomp>.<lambda> at 0x000001B5382F5840>, 
<function fun1.<locals>.<listcomp>.<lambda> at 0x000001B538BF3510>, 
<function fun1.<locals>.<listcomp>.<lambda> at 0x000001B538BF3598>]

[6, 6, 6, 6]
"""
```



### 闭包Closure

闭包的定义：如果在一个内部函数里，对在外部函数内（但不是在全局作用域）的变量进行引用，那么内部函数就被认为是闭包 (closure)

函数嵌套/闭包中的作用域：

```python
a = 1
def external():
    global a
    a = 200
    print a

    b = 100
    def internal():
        # nonlocal b
        print b  # Unresolved reference 'b' , 需要使用nonlocal引用b
        b = 200
        return b

    internal()
    print b
    
print external()

"""
200
100
200
None
"""
```

一样会报错- 引用在赋值之前，Python3有个关键字nonlocal可以解决这个问题，但在Python2中还是不要尝试修改闭包中的变量。
关于闭包中还有一个坑：

```python
from functools import wraps


def wrapper(log):
    def external(F):
        @wraps(F)
        def internal(**kw):
            if False:
                log = 'modified'
            print(log)

        return internal

    return external


@wrapper('first')
def abc():
    pass


print(abc())

# UnboundLocalError: local variable 'log' referenced before assignment
```

也会出现 引用在赋值之前 的错误，原因是解释器探测到了 if False 中的重新赋值，所以不会去闭包的外部函数（Enclosing）中找变量，但 if Flase 不成立没有执行，所以便会出现此错误。除非你还需要 else: log='var'  或者 if True 但这样添加逻辑语句就没了意义，所以尽量不要修改闭包中的变量。

好像用常规的方法无法让闭包实现计数器的功能，因为在内部进行 count +=1 便会出现 引用在赋值之前 的错误，解决办法： 

```python
def counter(start):
    count = [start]

    def internal():
        count[0] += 1
        return count[0]

    return internal


count = counter(0)
for n in range(10):
    print(count())  # 1,2,3,4,5,6,7,8,9,10

count = counter(0)
print(count())  # 1

```

由于 list 具有可变性，而字符串是不可变类型。



### nonlocal 声明扩大查找范围：嵌套的父级函数内

**实例**：

```python
count = 0


def outer():
    count = 10

    def inner():
        nonlocal count
        count = 20
        print("inner()_1:", count)

        def inner2():
            nonlocal count
            count = 30
            print("inner2():", count)

        inner2()
        print("inner()_2:", count)

    inner()
    print("outer():", count)


outer()
print(count)

"""
inner()_1: 20
inner2(): 30
inner()_2: 30
outer(): 30
0
"""
```

**实例**：

```python
def outer():
    count = 10

    def inner():
        count = 20
        nonlocal count
        print(count)

    inner()
    print(count)


outer()

"""
SyntaxError: name 'count' is assigned to before nonlocal declaration
"""
```



### global 声明：访问已有全局变量或新定义全局变量

**实例**：

```python
name = 0
def readonly():
    name = 10
    print("local:", name)

def readwrite():
    global name
    name = 20
    print("global:", name)
    global new_name
    new_name = 30

readonly()
print(name)

readwrite()
print(name)
print(new_name)

"""
local: 10
0
global: 20
20
30
"""
```



### locals() 和 globals()

**globals()**

==global 和 globals() 是不同的，global 是关键字用来声明一个局部变量为全局变量。globals() 和 locals() 提供了基于字典的访问全局和局部变量的方式==

比如：如果函数1内需要定义一个局部变量，名字另一个函数2相同，但又要在函数1内引用这个函数2。

```python
def var():
    pass


def f2():
    var = 'Just a String'
    f1 = globals()['var']
    print(var)
    return type(f1)


print(f2())

"""
Just a String
<class 'function'>
"""
```

**locals()**
如果你使用过Python的Web框架，那么你一定经历过需要把一个视图函数内很多的局部变量传递给模板引擎，然后作用在HTML上。虽然你可以有一些更聪明的做法，还你是仍想一次传递很多变量。先不用了解这些语法是怎么来的，用做什么，只需要大致了解locals()是什么。
可以看到，==locals() 把局部变量都给打包一起扔去了==。

```python
@app.route('/')
def view():
    user = User.query.all()
    article = Article.query.all()
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    s = 'Just a String'
    return render_template('index.html', user=user, article=article, ip=ip, s=s)
    # 或者 return render_template('index.html', **locals())
```

文章知识点与官方知识档案匹配，可进一步学习相关知识
