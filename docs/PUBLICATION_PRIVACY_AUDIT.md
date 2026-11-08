# GitHub公开发布材料与隐私审计

日期：2026-11-08。审计范围：`D:/codex学术论文2` 与 `D:/Codex工作区/busbra_sam_pilot`；目标由主代理确认是公开仓库。本审计仅检查目录、文本、结构化字段、文件大小、代码生成逻辑和工作簿容器，没有打开或显示患者图像/参考掩膜/预测掩膜像素，也没有执行网络写入。

## 结论

可以发布代码、论文规划、科学方案、经审阅的试验记录、伪名Case级数值、汇总图、模型策略JSON和版本信息。不能直接把两个项目目录整体上传。尤其当前工作台的旧CSV/XLSX/SQLite仍含官方原始身份编号和绝对病例路径；部分所谓Figure_3实际含逐例预测掩膜，必须排除。

IACS_v2核心开发数据库已经伪名化，应该优先保留其原字节或可精确还原的gzip，以保持训练输入哈希和冻结模型的可复现性，不应机械删除所有包含`path`或`ID`的字段。

## 1. IACS_v2冻结输入：可以保真发布

实查文件：`research_output/IACS_v2/dev_20261108_123935/metrics/sensitivity_samples.json`。

- 大小：5,753,009字节。
- SHA-256：`671c1d1eb7323d8480993c8c5b2e286836dc72b802a01be04582d0ee5d504e47`。
- 包含213个Case、213图、1491个候选条件记录。
- 递归检查没有`patient_id/case_id/CaseID/image_id/lesion_id`原始身份字段，没有绝对路径；所有`case_key/image_key`符合24–64位十六进制HMAC格式。
- 1491处`prediction_mask`、1278处`processed_image`都是安全的匿名相对artifact引用，无绝对路径或父目录跳转。保留这些字符串不等于上传像素文件；公开README应明确对应图像资产未包含。
- 输入/参考/预测SHA、模型权重SHA、提示框和评价数值属于追溯信息。不要为了隐藏不存在的原始ID而改写此JSON，否则将破坏`training_sample_hash`复现。

可一并保真发布的配套数值材料：`features/`、`metrics/`、`cross_validation/`、`statistics/`内的匿名JSON/CSV及策略模型；`fold_assignments.csv`和策略内HMAC Case列表可以保留，不能发布HMAC密钥或原身份对照表。

`compression/`下另外1491个任务JSON共4,165,890字节，逐个递归扫描没有原始身份字段、非HMAC键或绝对路径，可以仅发布这些JSON而排除相邻PNG/JPEG。不要按整个目录递归复制。

## 2. 数值报告与科研汇总图

`iacs_v2/reporting.py`明确只读取数值结果、不读取患者图像。其8种图为流程、特征分布、压缩质量、OOF比较、特征决策、折间表现、亚组和数值Pareto图。因此IACS_v2的`reports/publication*/figure_01–08.{png,svg,pdf}`可作为已知来源的汇总图候选。

报告导出的`case_`/`image_`加16位SHA键是对已有HMAC再散列的展示伪名；它们不是原始CaseID。自动扫描器将其标为“非纯hex键”属于已核实的假阳性。

3个IACS工作簿均无`xl/media`嵌入图像：

- `reports/publication/IACS_v2_results.xlsx`
- `reports/publication_v2/IACS_v2_results.xlsx`
- `tables/IACS_v2_SCI_results.xlsx`

可以发布数值工作簿，但应让README区分展示伪名与冻结输入HMAC命名空间。

大报告存在重复：`report.json`约50.9MB，各两份；HTML约46.1MB和20.0MB。建议保留完整数值JSON的gzip、一个最新展示工作台及版本/哈希清单；重复文件可用清单说明来源，不必复制多个相同50MB快照。gzip必须能还原原始JSON字节，公开保存未压缩SHA-256。

## 3. 旧smoke两轮

两轮`smoke_20261108_v1/v2`各102任务，对应相同3个Case，不能计为6个独立样本。结构化扫描没有原始身份字段，任务JSON使用HMAC和匿名相对路径。

可发布：任务结果JSON、事件日志、运行快照、匿名CSV/XLSX、数值summary图与说明。`run.json`内有少量机器级数据/模型根路径；不属于逐例身份映射，但对公开可移植配置可另制示例副本。不要修改原运行快照后仍声称原哈希未变。

必须排除：`processed*`、`prediction*`等逐例像素文件，以及任何封装了这些文件的旧ZIP。不要仅依据扩展名认为PNG/PDF/SVG一定是科研汇总图。

## 4. 当前工作台：必须制作publication-only副本

`D:/codex学术论文2/research_output`的实查风险：

| 材料 | 发现 | 发布处理 |
|---|---|---|
| `dataset_report/inventory.json` | 2131处patient_id、2147处image_id、5026处绝对路径值 | 不原样发布；只导出匿名Case级数值和聚合盘点 |
| 实验`config.json`的records | 原patient_id/image_id、原图/参考路径 | 为公开版记录生成HMAC键，删去原图/参考路径；保留原冻结配置的哈希作为来源 |
| `results.jsonl`与`tasks.sqlite3` | 原patient_id/image_id；image_key为可回查的dataset:原图编号；逐例绝对路径 | 从原库只读导出到全新匿名JSON/CSV；原SQLite不上传 |
| `per_image.csv`、`per_patient_case.csv`及Table2/4/5 | 仍带原身份列，部分表有绝对病例路径 | 重新渲染匿名副本，不能直接复用现成导出 |
| 现成XLSX | 由原逐例表生成，仍保留身份列 | 从匿名数据重新生成，而非复制原工作簿 |
| 失败记录error全文 | 可能含文件路径/原任务编号 | 保留error_type、stage、status和计数；清理全文中的原始路径/身份 |

建议使用仅在本机保存的新32字节发布盐，按`publication-v1 / dataset / identity_type / original_value`生成稳定HMAC。相同数据集的同一Case跨旧实验使用相同发布键，Case和image采用不同域；不同数据集不因数值编号相同而合并。原始编号→发布键映射和盐不进入仓库。

已冻结IACS_v2数据库保持原HMAC以维持复现；旧工作台的发布命名空间另行注明。不要把新发布HMAC与冻结输入HMAC混称同一键，也不要在公开仓库附上其私有桥接表。

如果确实需要公开SQLite，必须用脱敏记录创建全新数据库；不要复制原库后UPDATE/DELETE，因为旧页面/事件表仍可能残留原始值。首选可审计的JSON/CSV。

## 5. 图像排除项：已核实的隐藏风险

- `research_platform/exports.py::_figures`生成`Figure_3_observed_prediction.png`，内容是逐例预测mask，且印有原image_id；即使位于exports/figures，也必须排除。
- `scripts/create_observed_panels.py`明确读取处理图、原图、人工参考和预测掩膜；`research_output/figures/observed_panels/Figure_2_observed_compression.*`与`Figure_3_observed_segmentation.*`全部排除，包含PNG、SVG、PDF三个版本。SVG/PDF可能嵌入患者像素。
- `research_output/figures/workbench_export.jpg`是未按已知生成逻辑证明无患者像素的截图，暂不发布。
- 原pilot的`results/contour_comparison.png`是逐例轮廓对照，排除。
- IACS_v2的`compression/`、`deployment/`及CPU smoke中的JPEG/PNG都是逐例资产，排除；只保留审阅后的JSON数值记录。

## 6. 必须排除的目录/文件类别

- `.compression_private/`、所有`pseudonym.key`、新发布盐/身份映射、任何凭据或连接配置。
- `.venv/`、`.venv-compression/`、wheels、缓存、编译文件、浏览器缓存、PID/锁文件。
- 原始图像、人工参考、逐例压缩图、逐例预测mask、模型权重、临床工作簿和原始数据压缩包。
- 原pilot的`breast_external_raw/`、`breast_external_extracted/`和原始`breast_external_test/manifest.csv`不能整体上传；该目录含临床表格、图像和原始身份清单。
- 旧稿件/投稿包、隐藏作者/单位工作目录、旧备份及不透明ZIP不自动纳入。它们可能有患者图、作者个人信息或嵌入媒体，应另做明确范围的审阅，不能借“论文规划全部上传”跳过检查。

## 7. 扫描规模与限制

在跳过环境、原始数据目录、投稿包和逐例像素目录后，扫描586个文本/结构化候选文件，共253,390,901字节（约241.65MiB）：309 JSON、7 JSONL、79 CSV、49 log、48 Markdown及代码/HTML等。另专门审查IACS_v2的1491个任务JSON共约3.97MiB。

自动扫描发现33个“原身份或非纯hex键”候选，其中IACS报告10个是已核实的展示伪名假阳性；其余23个确实需要脱敏或排除，主要来自旧工作台和原pilot旧研究逐例表。100个文件有绝对路径（含机器级路径，不全是逐例敏感路径）；应按字段含义区分，不能一律破坏冻结输入。

没有发现所检查文本中的常见GitHub PAT、AWS访问键或私钥标记。此结论仅限本次明确模式，不等于所有凭据类型均已穷尽。发布前仍需对最终staging树再扫描，检查没有漏带身份映射、病历像素、临床表格、源数据库和环境目录。

细项扫描结果保存于同目录`github_publish_privacy_findings.json`，只含字段计数/相对文件名，不包含原始患者身份值；不应将其当成自动全量上传白名单。
