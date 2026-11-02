# 代码

本目录保存当前论文的实验实现、所需依赖、测试和本地复核工具。先读[研究方案](../docs/研究方案.md)，再按下面的顺序阅读代码。

| 位置 | 用途 |
| --- | --- |
| [revision_20261022/](revision_20261022/) | 冻结主实验：场景生成、语言接口、校准、物理核验及模型推理 |
| [key_validation/](key_validation/) | 搜索、证书、独立参考与决策审计 |
| [audit/](audit/) | 有理数校准、独立物理检查、共享证书、停止规则与 H 场景变换 |
| [project/](project/) | 数据解包、材料检查、结果重放、统计、绘图和论文构建 |
| [project/manifest.json](project/manifest.json) | 全项目交接材料的文件大小与哈希校验清单 |
| [tests/](tests/) | 决策规则、物理边界、冻结流程及数据包检查 |
| [scripts/kbs_project.sh](scripts/kbs_project.sh) | 本地复核统一入口 |
| [requirements-analysis.txt](requirements-analysis.txt) | 本地分析环境依赖 |
| [configs/experiment_resources.json](configs/experiment_resources.json) | 资源配置；规划器为 `experiment_resources.py` |
| [source_bindings.json](source_bindings.json) | 冻结源码来源、当前哈希及适配说明 |

其他 Python 包是上述模块实际导入的依赖，保留其原有名称。阅读主流程时，建议依次查看 `language_interface.py`、`runtime_checks.py` 和 `language_analysis.py`。

以下命令在项目根目录执行。核对仓库材料及已下载外部包的完整性；尚未下载的外部包会单独列出：

```bash
bash code/scripts/kbs_project.sh verify
```

按[复核说明](../docs/复核说明.md)配置环境，并按[数据下载说明](../data/README.md#原始数据下载)准备 CPU 和主语言包后，执行完整本地复核：

```bash
KBS_PYTHON=work/venv/bin/python bash code/scripts/kbs_project.sh all
```

结果写入根目录下的 `work/`。这些命令不派发模型推理；服务器接续须按[工作计划](../docs/工作计划.md)核对请求与进程，成功请求不重复运行。科学设置的变更应使用新的实验批次。

[返回项目说明](../README.md)
