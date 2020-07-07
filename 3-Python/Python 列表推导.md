## 列表推导（list comprehensions）

> 场景1：将一个三维列表中所有一维数据为a的元素合并，组成新的二维列表。

最简单的方法：新建列表，遍历原三维列表，判断一维数据是否为a，若为a,则将该元素append至新列表中。
缺点：代码太繁琐，对于Python而言，执行速度会变慢很多。
针对场景1，我们首先应该想到用**列表解析式**来解决处理，一行代码即可解决：

```python
lista = [item for item in array if item[0] == 'a']
```

那么，何为列表解析式？
官方解释：列表解析式是Python内置的非常**简单**却**强大**的可以用来创建list的生成式。

**强大具体如何体现？**
可以看到，使用列表解析式的写法更加简短，除此之外，因为是Python内置的用法，底层使用C语言实现，相较于编写Python代码而言，运行速度更快。

> 场景2: 对于一个列表，既要遍历索引又要遍历元素。

这里可以使用Python内建函数enumerate，在循环中更好的获取获得索引。

```python
array = ['I', 'love', 'Python']
for i, element in enumerate(array):
    array[i] = '%d: %s' % (i, seq[i])
```

可以使用列表推导式对其进行重构：

```python
def getitem(index, element):
    return '%d: %s' % (index, element)

array = ['I', 'love', 'Python']
arrayIndex = [getitem(index, element) for index, element in enumerate(array)]
```

据说这种写法更加的Pythonic。

> 总结：如果要对现有的可迭代对象做一些处理，然后生成新的列表，使用列表推导式将是最便捷的方法。

