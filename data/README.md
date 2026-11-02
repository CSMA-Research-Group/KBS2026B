# 数据

本目录保存论文所需的冻结协议、校验信息、图表数据及原始记录包入口。CPU、主语言记录包从 OneDrive 按需下载，本地文件由 Git 忽略；Experiment H、S1 和逐文件校验清单随仓库提供。原始记录保留不动，新的分析结果写入根目录下的 `work/`。

| 位置 | 内容 |
| --- | --- |
| [archives/](archives/) | Experiment H、S1、逐文件校验清单，以及 CPU、主语言包的本地下载位置 |
| [protocols/](protocols/) | 冻结语言与 CPU 协议、推理环境和预检记录 |
| [models/](models/) | 模型文件的标识与哈希，不含模型权重 |
| [verification_cost/](verification_cost/) | E5/E7 成本诊断、逐实例记录及校准阈值 |
| [illustrations/](illustrations/) | 论文示意图定义、CPU 绘图点及来源记录 |
| [manifest.json](manifest.json) | 四个记录包的大小、哈希、成员数量、解包位置及外部下载链接 |
| [selection.json](selection.json) | 已选用记录的来源、哈希和筛选位置 |

## 原始数据下载

打开对应共享链接，下载文件并按下表保存。大小按 1 MB = 1,000,000 字节计算。

| 数据包 | 共享下载链接 | 大小 | 保存位置（相对项目根目录） |
| --- | --- | --- | --- |
| CPU | [OneDrive 下载](https://1drv.ms/u/c/866abf2c283fa755/IQA0BOpEsxffQ44XQJwWZFA-AW8m8VvyptENUzLVy_HHPJE?e=vzhfpA) | 127.79 MB | `data/archives/cpu.tar.gz` |
| 主语言 | [OneDrive 下载](https://1drv.ms/u/c/866abf2c283fa755/IQB6p9EXt_2lTLZpYNe2mkv1AWyYKWFD0-NGOYF6vA-Z_7E?e=PUth3s) | 346.44 MB | `data/archives/language.tar.gz` |

两个包的准确字节数和 SHA-256 固定在 [manifest.json](manifest.json)。下载后可运行 `bash code/scripts/kbs_project.sh verify` 校验已有文件；未下载的外部包会单独列出。需要运行相应实验复核时，再下载对应包；完整 `check` / `all` 需要两个包齐备。`build` 编译论文无需下载实验包。

在项目根目录运行以下命令，可校验并单独解包已下载的 CPU 数据：

```bash
bash code/scripts/kbs_project.sh prepare cpu
```

也可选择 `language`、`experiment_h` 或 `s1`。程序先校验大小与 SHA-256，再逐文件校验并解包到 `work/evidence/`。缺少外部包时会提示共享链接与保存位置，不自动下载。

当前主语言测试响应不完整，H 只有准备好的输入和检查记录，尚无本地模型响应。S1 属于早期 oracle 接口分析，E5/E7 属于局部快照诊断。各类材料的用途和统计单位见[数据说明](../docs/数据说明.md)，完成状态见[工作计划](../docs/工作计划.md)。

[返回项目说明](../README.md)
