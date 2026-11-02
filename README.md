# KBS2026B

投稿目标：**Knowledge-Based Systems**。

本研究讨论：语言解释存在不确定性时，哪些语义与物理证据足以支持执行、澄清或拒绝，以及获得这些证据需要多少核验工作。

从 [研究方案](docs/研究方案.md)、[论文 PDF](latex/main.pdf) 和 [参与说明](docs/参与说明.md) 开始。当前主语言实验尚未完成，Experiment H 已备好输入；下一步范围见 [工作计划](docs/工作计划.md)。

| 目录 | 内容 |
| --- | --- |
| [latex/](latex/README.md) | 当前论文源码、图、表、参考文献和模板 |
| [code/](code/README.md) | 实验实现、必要求解器依赖、复核入口和测试 |
| [data/](data/README.md) | 实验记录包及下载入口、协议、模型文件哈希及图表数据 |
| [docs/](docs/README.md) | 研究方案、参与说明、工作计划、数据说明和复核方法 |
| `work/` | 运行时生成的解包缓存、分析结果和构建输出 |

CPU 和主语言原始记录包通过 [OneDrive 共享链接](data/README.md#原始数据下载)按需下载，保存到 `data/archives/`；本地文件由 Git 忽略。Experiment H、S1 和校验清单随仓库提供。

环境准备见 [复核说明](docs/复核说明.md)。依赖及所需数据包齐备后，在本目录运行：

```bash
KBS_PYTHON=work/venv/bin/python bash code/scripts/kbs_project.sh all
```

该命令依次核对数据和论文表格、运行测试、重放成本诊断、导出图并编译论文。需要的原始记录会自动解包至 `work/evidence/`，不调用模型。

实验文件保持原始内容，并有逐文件校验值。完整主实验、H 真实推理与完整流程成本比较仍需完成；已有核查结果及其范围见 [复核说明](docs/复核说明.md)。

[code/project/manifest.json](code/project/manifest.json) 记录交接材料的文件大小和哈希，用于核对完整性。各目录的 README 提供本目录入口，研究安排统一见 `docs/`。
