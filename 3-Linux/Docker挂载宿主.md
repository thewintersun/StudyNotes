# Docker 挂载宿主机文件目录

docker可以支持把一个宿主机上的目录挂载到镜像里。

`docker run -it -v /home/dock/Downloads:/usr/Downloads ubuntu64 /bin/bash`
 通过-v参数，冒号前为宿主机目录，必须为绝对路径，冒号后为镜像内挂载的路径。

现在镜像内就可以共享宿主机里的文件了。

默认挂载的路径权限为读写。如果指定为只读可以用：ro
 `docker run -it -v /home/dock/Downloads:/usr/Downloads:ro ubuntu64 /bin/bash`



docker还提供了一种高级的用法。叫数据卷。

数据卷：“其实就是一个正常的容器，专门用来提供数据卷供其它容器挂载的”。感觉像是由一个容器定义的一个数据挂载信息。其他的容器启动可以直接挂载数据卷容器中定义的挂载信息。

看示例：
 `docker run -v /home/dock/Downloads:/usr/Downloads --name dataVol ubuntu64 /bin/bash`
创建一个普通的容器。用--name给他指定了一个名（不指定的话会生成一个随机的名子）。

再创建一个新的容器，来使用这个数据卷。
 `docker run -it --volumes-from dataVol ubuntu64 /bin/bash` 

--volumes-from用来指定要从哪个数据卷来挂载数据。

 