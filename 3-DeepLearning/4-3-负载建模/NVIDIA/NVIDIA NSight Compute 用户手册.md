## NSight Compute 用户手册

非交互式配置文件活动             

从NVIDIA Nsight Compute启动目标应用程序             

启动NVIDIA Nsight Compute时，将出现欢迎页面。单击快速启动打开连接对话框。如果未显示“连接”对话框，则可以使用主工具栏上的“连接”按钮打开它，只要当前未连接。从Connection下拉列表中选择左侧的目标平台和本地主机。然后，填写launch details并选择launch。在“活动”面板中，选择“概要文件”活动以启动预配置概要文件会话的会话，并启动命令行探查器以收集数据。提供输出文件名以允许使用启动按钮启动会话。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108164951104-1693345397.png)

  其他启动选项             

有关这些选项的详细信息，请参阅命令行探查器的命令行选项。这些选项被分组到选项卡中：Filter选项卡公开选项来指定应该分析哪些内核。选项包括内核regex过滤器、要跳过的启动次数以及要评测的启动总数。Section选项卡允许您为每次内核启动选择应该收集的部分。采样选项卡允许您为每次内核启动配置采样选项。另一个选项卡包含通过--metrics选项收集NVTX信息或自定义度量的选项。              

Section选项卡允许您为每次内核启动选择应该收集的部分。将鼠标悬停在某个部分上，以查看其作为工具提示的说明。要更改默认启用的节，请使用“Sections/Rules信息”工具窗口。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165043740-1998296823.png)

有关此活动中可用选项的详细说明，请参阅配置文件活动。             

浏览报表                      

默认情况下，profile报告会出现在Details页面上。您可以在报表的不同报表页之间切换，报表左上角的下拉列表标记为“页”。报告可以包含任意数量的内核启动结果。启动下拉列表允许在报告中的不同结果之间切换。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165138766-175160666.png)

区分多个结果             

在“详细信息”页面上，按“添加基线”按钮以将当前结果提升为焦点，以将来自此报表的所有其他结果以及在同一个NVIDIA Nsight Compute实例中打开的任何其他报告进行比较。如果设置了基线，详细信息页面上的每个元素都会显示两个值：焦点中结果的当前值和基线的相应值或与相应基线值相比的更改百分比。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165206996-1054465435.png)

 使用下拉按钮、配置文件菜单或相应工具栏按钮中的清除基线条目删除所有基线。有关更多信息，请参见基线。             

执行规则             

在“详细信息”页面上，某些部分可能会提供规则。按Apply按钮执行单个规则。顶部的Apply Rules按钮执行焦点中当前结果的所有可用规则。规则也可以是用户定义的。有关详细信息，请参阅《自定义指南》。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165304245-1899864221.png)

连接对话框             

使用“连接”对话框启动并附加到本地和远程平台上的应用程序。首先选择要分析的目标平台。默认情况下（如果支持），将选择本地平台。选择要在其上启动目标应用程序或连接到正在运行的进程的平台。             

连接对话框

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165430403-417185653.png)

使用远程平台时，将要求您在顶部下拉列表中选择或创建连接。要创建新连接，请选择+并输入连接详细信息。使用本地平台时，将选择localhost作为默认值，不需要进一步的连接设置。如果分析将在同一平台的远程系统上进行，则仍然可以创建或选择远程连接。             

根据您的目标平台，选择Launch或Remote Launch来启动应用程序以在目标上进行分析。请注意，只有在目标平台支持时，远程启动才可用。             

为应用程序填写以下启动详细信息：             

应用程序可执行文件：指定要启动的根应用程序。请注意，这可能不是您希望评测的最终应用程序。它可以是创建其他进程的脚本或启动器。             

工作目录：应用程序将在其中启动的目录。             

命令行参数：指定要传递给应用程序可执行文件的参数。             

环境：为启动的应用程序设置的环境变量。             

选择“附加”将探查器附加到已在目标平台上运行的应用程序。此应用程序必须已使用另一个NVIDIA Nsight Compute CLI实例启动。该列表将显示目标系统上运行的所有可附加的应用程序进程。选择“刷新”按钮以重新创建此列表。             

最后，为启动或附加的应用程序选择要在目标上运行的活动。请注意，并非所有活动都必须与所有目标和连接选项兼容。目前，存在以下活动：             

交互式配置文件活动             

配置文件活动             

远程连接             

支持SSH的远程设备也可以在连接对话框中配置为目标。要配置远程设备，请确保选择了支持SSH的目标平台，然后按+按钮。将显示以下配置对话框。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108165453450-2138985759.png)

NVIDIA Nsight Compute支持密码和私钥身份验证方法。在此对话框中，选择身份验证方法并输入以下信息：             

密码             

IP/主机名：目标设备的IP地址或主机名。             

用户名：用于SSH连接的用户名。             

Password：用于SSH连接的用户密码。             

端口：用于SSH连接的端口。（默认值为22。）              

部署目录：目标设备上用于部署支持文件的目录。指定的用户必须对此位置具有写入权限。             

私钥

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108171930889-417632961.png)

 IP/主机名：目标设备的IP地址或主机名。             

用户名：用于SSH连接的用户名。             

SSH私钥：用于向SSH服务器进行身份验证的私钥。             

SSH-Key-Passphrase：您的私钥的密码短语。             

部署目录：目标设备上用于部署支持文件的目录。指定的用户必须对此位置具有写入权限。             

输入所有信息后，单击“添加”按钮以使用此新连接。             

当在连接对话框中选择远程连接时，应用程序可执行文件浏览器将使用配置的SSH连接浏览远程文件系统，允许用户选择远程设备上的目标应用程序。             

在远程设备上启动“活动”时，将执行以下步骤：             

命令行探查器和支持文件将复制到远程设备上的部署目录中。（仅复制不存在或过期的文件。）             

应用程序可执行文件在远程设备上执行。             

对于交互式概要文件活动，将建立到远程应用程序的连接，并开始分析会话。              

对于非交互式概要文件活动，远程应用程序在命令行探查器下执行，并生成指定的报告文件。             

对于非交互式分析活动，生成的报告文件将复制回主机并打开。             

每个步骤的进度都显示在进度日志中。             

进度日志

请注意，一旦远程启动了任一活动类型，就可以在远程设备上的部署目录中找到进一步分析会话所需的工具。             

交互式配置文件活动             

交互式概要文件活动允许您启动控制目标应用程序执行的会话，类似于调试器。您可以单步执行API调用和工作负载（CUDA内核），暂停和恢复，并以交互方式选择感兴趣的内核和要收集的度量。             

此活动当前不支持分析或附加到子进程。             

支持NVTX             

收集应用程序或其库提供的NVTX信息。需要支持单步执行到特定的NVTX上下文。             

禁用分析启动/停止             

忽略应用程序对cu（da）ProfilerStart或cu（da）ProfilerStop的调用。             

从一开始启用分析             

从应用程序启动时启用分析。如果应用程序在第一次调用此API之前调用cu（da）ProfilerStart和内核，则禁用此选项非常有用。请注意，禁用此选项不会阻止您手动分析内核。              

缓存控制

控制分析期间GPU缓存的行为。允许的值：对于Flush All，在评测期间的每个内核重播迭代之前，都会刷新所有GPU缓存。虽然应用程序的执行环境中的度量值可能稍有不同而不会使缓存失效，但此模式在重播过程中以及在目标应用程序的多个运行中提供了最可复制的度量结果。             

对于Flush None，在分析期间不刷新GPU缓存。如果度量收集只需要一个内核重播过程，这可以提高性能并更好地复制应用程序行为。然而，一些度量结果将根据先前的GPU工作以及在重放迭代之间变化。这可能导致度量值不一致和越界。             

时钟控制             

控制分析期间GPU时钟的行为。允许值：对于基频，GPC和内存时钟在配置期间被锁定到各自的基频。这对热节流没有影响。对于None，在分析期间不会更改GPC或内存频率。             

配置文件活动             

Profile活动提供了一个传统的、可预先配置的profiler。在配置了要评测的内核、要收集的度量等之后，应用程序将在分析器下运行，而无需交互控制。一旦应用程序终止，活动即完成。对于通常不会自行终止的应用程序，例如交互式用户界面，您可以在分析完所有预期的内核之后取消该活动。             

此活动不支持附加到以前通过NVIDIA Nsight Compute启动的进程。这些进程将在“附加”选项卡中显示为灰色。

输出文件             

应存储收集的配置文件的报表文件的路径。如果不存在，则自动添加报表扩展名.ncu rep。文件名组件支持占位符%i。它被一个按顺序递增的数字替换，以创建一个唯一的文件名。这将映射到--export命令行选项。             

强制覆盖             

如果设置，则覆盖现有报告文件。这将映射到--force overwrite命令行选项。             

目标流程             

选择要分析的进程。仅在应用程序模式下，只分析根应用程序进程。在模式all中，将分析根应用程序进程及其所有子进程。这将映射到--target processes命令行选项。             

重播模式             

选择多次重放内核启动的方法。在模式内核中，单个内核的启动在目标应用程序的单个执行期间被透明地回放。在模式应用程序中，整个目标应用程序将被多次重新启动。在每次迭代中，为目标内核启动收集额外的数据。应用程序回放要求程序的执行是确定性的。这将映射到--replay模式命令行选项。有关重播模式的更多详细信息，请参阅内核评测指南。             

其他选项             

所有剩余的选项都映射到它们的等效命令行探查器。有关详细信息，请参阅NVIDIA Nsight Compute CLI文档中的“命令行选项”部分。             

重置             

“连接”对话框中的条目将保存为当前项目的一部分。在自定义项目中工作时，只需关闭项目即可重置对话框。             

不在自定义项目中工作时，条目将作为默认项目的一部分存储。通过关闭NVIDIA Nsight Compute，然后从磁盘中删除项目文件，可以从默认项目中删除所有信息。

 ![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108173817915-420944083.png)

 

主菜单             

文件             

新建项目使用“新建项目”对话框创建新的分析项目

连接             

连接打开连接对话框以启动或附加到目标应用程序。已连接时禁用。             

断开与当前目标应用程序的断开连接，允许应用程序正常继续并可能重新连接。             

终止断开连接并立即终止当前目标应用程序。             

调试             

暂停在下一次截获的API调用或启动时暂停目标应用程序。             

继续恢复目标应用程序。             

单步执行当前API调用或启动到下一个嵌套调用（如果有）或后续API调用，否则。             

单步执行跳过当前API调用或启动，并在下一次非嵌套API调用或启动时挂起。              

单步执行跳出当前嵌套的API调用或启动到下一个非父API调用或启动上一个级别。             

冻结API

禁用时，启用所有CPU线程并在单步执行或继续执行期间继续运行，并且至少有一个线程到达下一个API调用或启动时，所有线程都将停止。这也意味着，在单步执行或继续执行期间，当前选定的线程可能会发生更改，因为旧的选定线程没有前进的进程，并且API流会自动切换到具有新API调用或启动的线程。启用时，仅启用当前选定的CPU线程。所有其他线程都被禁用和阻止。             

如果当前线程到达下一个API调用或启动，则单步执行现在完成。所选线程从不更改。但是，如果所选线程没有调用任何进一步的API调用，或者在某个屏障处等待另一个线程取得进展，则单步执行可能无法完成并无限期挂起。在这种情况下，请暂停，选择另一个线程，然后继续单步执行，直到原始线程被解除阻塞为止。只会在这个模式下前进。             

中断API错误启用后，在恢复或单步执行期间，一旦API调用返回错误代码，执行将被挂起。             

运行到下一个内核请参阅API流工具窗口。             

运行到下一个API调用请参阅API流工具窗口。             

运行到下一个范围开始查看API流工具窗口。             

运行到下一个范围结束参见API流工具窗口。             

API统计打开API统计工具窗口             

API流打开API流工具窗口             

资源打开资源工具窗口             

NVTX打开NVTX工具窗口

简况             

配置文件内核在内核启动时挂起时，请使用当前配置选择配置文件。              

自动配置文件启用或禁用自动配置文件。如果启用，则将使用当前节配置分析与当前内核筛选器（如果有）匹配的每个内核。             

清除基线清除所有当前基线。             

节/规则信息打开节/规则信息工具窗口。             

工具             

Project Explorer将打开Project Explorer工具窗口。             

输出消息打开输出消息工具窗口。             

选项打开“选项”对话框。              

窗口             

保存窗口布局允许您指定当前布局的名称。布局以“.nvlayout”文件的形式保存到documents目录中的layouts文件夹中。             

应用窗口布局保存布局后，可以使用“应用窗口布局”菜单项恢复它们。只需从子菜单中选择要应用的条目。             

管理窗口布局允许您删除或重命名旧布局。             

恢复默认布局将视图恢复到其原始大小和位置。             

显示欢迎页面打开欢迎页面。             

帮助             

文档打开NVIDIA Nsight Compute online的最新文档。             

Documentation（local）打开工具附带的NVIDIA Nsight Compute的本地HTML文档。             

检查更新联机检查是否有新版本的NVIDIA Nsight Compute可供下载。             

重置应用程序数据重置磁盘上保存的所有NVIDIA Nsight计算配置数据，包括选项设置、默认路径、最近的项目引用等。这不会删除保存的报告。             

发送反馈打开一个对话框，允许您发送bug报告和特性建议。可选地，反馈包括基本系统信息、屏幕截图或附加文件（如配置文件报告）。             

“关于”打开“关于”对话框，其中包含有关NVIDIA Nsight Compute版本的信息。

工具列             

主工具栏显示主菜单中常用的操作。有关它们的说明，请参见主菜单。

![img](https://img2020.cnblogs.com/blog/1251718/202011/1251718-20201108173538523-372259575.png)       

状态横幅             

状态横幅用于显示重要消息，例如探查器错误。单击“X”按钮可以解除消息。同时显示的横幅数量是有限的，如果出现新的消息，旧消息可以自动关闭。使用“输出”窗口查看消息。