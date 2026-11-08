# 开发计划

日期：2026-11-08。依据 PROJECT_AUDIT.md，采用现有Python技术栈，新增本地实验扩展；原项目只读。

| 阶段 | 工作与新增文件 | 进入下一阶段前的核查 |
|---|---|---|
| A/B | 定位联合项目、审查旧流程、PROJECT_AUDIT.md | 原路径、版本与数据状态有证据 |
| C | 本计划 | 复用点与新代码范围明确 |
| D | backups/ 下旧脚本/配置副本和SHA256清单 | 备份字节一致，原权重只记录哈希 |
| E | research_platform/datasets.py、test_datasets.py | 两库配对、解码、重复、患者语义与冻结划分检查 |
| F | models.py、geometry.py | 复用提示函数，核验SAM，真实开发样本原图推理和mask保存 |
| G | compression.py | JPEG、PNG、WebP/JPEG2000能力、缩放、处理顺序、位深和原图保护测试 |
| H | engine.py、store.py | 唯一任务、逐例日志、续跑、失败重试、配置锁定 |
| I | metrics.py | 解析样例验证交并/边界/空值约定 |
| J | features.py | 特征只读原图像素，不传参考轮廓 |
| K | adaptive.py | 明确研究用未验证规则；只允许开发数据校准/锁定，禁止外部调参 |
| L | statistics.py | 患者聚类、配对、Holm、混合模型和非劣阈值门禁测试 |
| M | server.py、web/index.html | 本地八页工作台，所有可操作按钮连接真实接口，无数据时显示未运行 |
| N | exports.py | CSV/Excel、PNG/SVG/PDF、报告均从本次数据库生成，缺项明示 |
| O | tests/、TEST_REPORT.md | 真实技术试跑与自动测试分开报告，不称正式验证 |
| P | USER_MANUAL.md、EXPERIMENT_PROTOCOL.md、SCI_RESULTS_REPORT.md | 启动命令可独立复现，已完成/待完成/失败/假设分开 |

每阶段完成后运行对应检查，保存 DEVELOPMENT_PROGRESS.md。阶段存在依赖时按依赖顺序验证；并行编写的独立模块不能提前宣称验收通过。

U-Net仅在患者来源与足够训练数据、隔离划分可确认后训练；当前无已验证U-Net代码/权重，先报告条件不足，不用随机未训练模型冒充跨模型验证。

首次实际运行限BUS-BRA开发清单的固定顺序小样本，覆盖原图/压缩/缩放/恢复/评价/导出。完整实验矩阵与外部验证须在本研究方案、患者语义、预设容忍界值和IACS参数冻结后执行，保留旧研究查看测试集的事实。
