# python添加模块路径的三种方法

**查看python的模块路径方法是**：

```python
import sys
print sys.path
```

这个就会打印出所有的模块路径。

下边是在这个python系统路径中加入新的模块路径的三种方法，

1、添加环境变量PYTHONPATH, python会添加此路径下的模块，在.bash_profile文件中添加如下类似行:
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/site-packages

2、在site-packages路径下添加一个路径配置文件,文件的扩展名为.pth,内容为要添加的路径即可。但如果本来python就没有添加上site-packages目录，那这个方法其实也是用不了的。

3、sys.path.append()函数添加搜索路径,参数值即为要添加的路径。

