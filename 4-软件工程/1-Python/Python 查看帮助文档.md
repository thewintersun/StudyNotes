# Python忘记函数用法又不能切屏怎么办

自从上次整理了编程考试可以用的[库函数清单](http://3ms.huawei.com/km/blogs/details/7610225?l=zh-cn)后，有不少小伙伴来问如果忘记函数怎么用了咋办，这么多函数记不住啊:(

这里介绍一下直接利用`leetcode`在线查看函数清单和说明的方法。



**注意：下面的语句要点"执行代码"查看stdout结果，不要点到"提交"了~**



## 查询可使用的模块(modules)清单

-Python 代码

```python
import sys

for m in sys.modules:
    print(m)
```

![img](http://image.huawei.com/tiny-lts/v1/images/e30652682f1d98524cc1_297x316.png@900-0-90-f.png)

## 查询模块的文档

-Python 代码

```python
# print(help({module_name}))
print(help(math))
```

![img](http://image.huawei.com/tiny-lts/v1/images/876b12682f1e797bf2a6_553x319.png@900-0-90-f.png)

如果文档太长打印不完整，可以直接查函数清单

注意：有下划线前缀的是私有方法，不能用的

-Python 代码

```python
# print(dir({module_name}))
print(dir(math))
```

![img](http://image.huawei.com/tiny-lts/v1/images/497f12682f1f5cb62a74_988x266.png@900-0-90-f.png)



## 查询函数的文档

-Python 代码

```python
# print(help({module.function}))
print(help(math.log))
```

![img](http://image.huawei.com/tiny-lts/v1/images/5e5222682f20ce05d8fb_527x304.png@900-0-90-f.png)

