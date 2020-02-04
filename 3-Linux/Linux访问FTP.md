> 以下命令基于 CentOS 7.0 环境，其他 Linux 环境的命令稍有出入，请注意调整。

1. 安装 lftp 命令（lftp相比于ftp命令的优势，在于它支持递归下载目录，而ftp只能下载文件，无法下载目录。）

   ```bash
   yum install -y lftp
   ```

2. 使用lftp命令访问 FTP 服务器

   ```bash
   # 方式一：交互式命令，适合手动执行
   #连接 FTP 服务器，认证通过后，可以通过help查看帮助，一般用cd进入目录，get下载文件。
   lftp ftp://dataset:123456@192.144.0.211 
   # get filename: 下载单个文件
   # mget *：批量下载多个文件
   # mirror directoryname: 递归下载整个目录
   
   # 方式二：非交互式命令，适合自动化执行
   # 查询FTP服务器下的目录结构。
   lftp -u dataset,123456 -e "ls /face; exit" 192.144.0.211 
   # 下载FTP服务器的文件夹，注意将目录修改为所需的目录。
   lftp -u dataset,123456 -e "mirror -c -P 10 -v /face/LFW/ /root/LFW/; exit" 192.168.47.206
   
   # 注意，使用 lftp 命令的 mirror 功能时，谨慎使用 "-e" 或 “--delete” 参数，该参数表示"删除远程目录上不存在的文件"。
   ```

3. 如需下载大文件，为避免对其他用户的影响，建议在非工作时间（午休、凌晨）下载。此时可以将lftp的下载 命令配置到 /etc/crontab 文件，使用定时下载。以下是 /etc/crontab 里的一个定时任务示例：

   ```bash
   # 以下定时任务表示在 12 月 25 日凌晨 01:00 发起下载。
   00 01 25 12 * root /usr/bin/lftp -u dataset,123456 -e "mirror -c -P 10 -v /face/LFW/ /root/LFW/; exit" dataset.inhuawei.com
   
   # 注意：定时任务完成后，记得清除 /etc/crontab 里的配置。
   ```

