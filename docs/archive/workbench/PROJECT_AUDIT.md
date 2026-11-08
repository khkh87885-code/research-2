# 项目审计：乳腺超声自适应压缩与分割可靠性

审计日期：2026-11-08（Asia/Shanghai）。状态：已定位并审阅既有代码，数据与运行环境复核完成；后续真实技术试跑见TEST_REPORT.md。本研究正式实验尚未开展。

## 1. 项目范围与审计证据

最初工作目录 `D:\codex学术论文2` 实查为空。用户随后指定既有资产根目录 `D:\codex论文`。定向只读检查还定位到与之直接关联的推理工程 `D:\Codex工作区\busbra_sam_pilot` 和官方 SAM 源码 `D:\Codex工作区\segment-anything`。三处联合构成现有研究资产，不应把空工作目录当作缺失整个项目的证据。

本次新增内容保存于 `D:\codex学术论文2`，作为既有推理工程的压缩实验扩展，读取既有数据、权重和提示框函数；原工程、投稿材料、旧实验锁文件和原始结果不改写。不复制或替换 SAM 权重，不重建既有数据。

根目录和相关源码目录未发现适用的 AGENTS.md。`D:\codex论文` 与 pilot 目录未发现 Git 仓库；官方 SAM 源码有 Git，当前提交为 `dca509fe793f601edb92606367a655c15ac00fdf`，工作树干净。

## 2. 架构与现有功能（20项检查）

| 检查项 | 实查结果 / 证据 |
|---|---|
| 项目根目录 | 数据/投稿资产位于 `D:\codex论文`；Python 推理脚本位于关联 pilot 目录 |
| 编程语言 | Python，旧推理环境 3.8.20 |
| 前端框架 | 未发现现有应用前端；投稿图片不属于工作台 |
| 后端框架 | 无 Web 后端，现有脚本为命令行流程 |
| 数据库 | 未发现研究数据库；CSV / JSON / PNG 保存 |
| 数据目录 | `D:\codex论文\BUSBRA`；pilot 下 `breast_external_extracted` 与 `breast_external_test` |
| BUS-BRA 读取 | `run_development_pilot.py:40` 读取原有 split CSV 的 image_path / mask_path；数据根来自 `run_single_case.py:17` |
| BrEaST 读取 | `run_formal_validation.py:235` 读取 manifest；`:89` verify_png_pairs 校验配对/哈希；原始工作簿与压缩包在 breast_external_raw |
| 人工轮廓 | BUSBRA/Masks/mask_*.png；BrEaST 按既有官方工作簿配对主病灶 PNG，另有非主病灶轮廓 |
| SAM 版本 | SAM v1，官方包 1.0，ViT-B；不替换为 SAM2 |
| 权重 | `D:\codex论文\models\sam_vit_b_01ec64.pth`，375042383 字节；SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| 推理入口 | run_single_case.py、run_development_pilot.py、run_full_development.py、run_formal_validation.py |
| 预处理 | BUS-BRA PIL 转 RGB；BrEaST RGBA 先合成到不透明黑底再转 RGB，见 rgb_over_black（formal runner :83） |
| 模型输出 | SamPredictor 返回 masks、scores、logits，单掩膜，原输入像素坐标；旧测量函数仅返回数值，不保存逐例掩膜 |
| 尺寸转换 | 旧主实验无压缩缩放；SAM 内部 ResizeLongestSide；旧绘图的显示缩放不等于实验处理 |
| 坐标映射 | prompt_boxes 由原始人工轮廓外扩10%，在原图坐标调用 predictor；旧代码没有压缩版本映射接口 |
| 评价指标 | Dice、预测 IoU、四方向提示扰动一致性、风险A/B、推理时间与预警统计 |
| 上传下载 | 未发现交互上传/下载端点 |
| 结果保存 | results、pilot_40、development_213、formal_validation 下 CSV/JSON/图；已有锁文件和校验记录 |
| 运行命令 | 旧入口可使用 `D:\Miniconda3\envs\learn_pytorch\python.exe -I <绝对脚本路径>`；旧部分入口拒绝覆盖已有输出，不能为验证而直接重跑原目录 |

## 3. 数据与科研状态

- BUS-BRA：实查1875张图与1875张同名对应轮廓；bus_data.csv 含1875个唯一 ID、1064个唯一 Case。现有开发清单213行，留出清单851行，每行一位所列患者；两清单患者和图像交集均为0，路径存在，patient_id 与 Case 的映射吻合。Case 的患者定义仍需来源文件核实；不通过文件名推断患者或病灶身份。
- BrEaST：既有清单列252个有主病灶的 CaseID；配对审计记载256例原图，4例正常图无主轮廓、14个其他轮廓，不能静默丢弃这些数据状态。CaseID 被旧清单命名为 patient_id，来源语义需要单独确认，不能仅凭列名宣称患者级独立。
- 既有配对审计记载跨库解码像素完全重复为0，并有黑底透明度补充核查。将重新检查实际文件；没有完全相同像素不能证明没有相同患者、近重复图或预训练集重合。
- BrEaST 31张原图含透明信息；8张在黑底合成后 RGB 与直接丢弃 alpha 不同。新实验必须沿用黑底合成，原始文件基线按字节保留，显示/模型输入的转换另外记录。
- 旧开发、内部留出和外部测试已用于另一项分割失效预警研究。旧实验数值不能当作本次压缩研究结果，数据也不能描述为研究人员从未查看过的测试集。
- 当前未核实两库患者来源独立性。暂用“跨数据集验证/外部数据集验证”，不称独立外部验证。

## 4. 模型与环境状态

现有 GPU 为 RTX 3060 Laptop，显存6144 MiB。现有 learn_pytorch 环境可定位 Python 3.8.20、torch/torchvision/Pillow/numpy/matplotlib。未使用隔离选项时，当前宿主的 base site-packages 污染使 typing_extensions 导入出错；优先用 `-I` 隔离运行并验证，不判定整个 Conda 环境损坏。基础 Python 3.14.7 不具备本项目科学计算依赖。

旧脚本要求 CUDA，尚无统一 CPU 路径。新适配器必须先核验源码提交与权重哈希，再 `torch.load(..., weights_only=True)`，记录设备/版本/同步计时，保存掩膜；机制实验只生成一次原图提示框，按实际整数输出尺寸映射；应用实验使用独立、不读人工轮廓的固定提示，标签明确标为未验证自动定位方法。

## 5. 可复用方案与问题

复用真实图像/轮廓、官方工作簿配对关系、冻结 split 和外部 manifest、SAM 源码/权重，以及原 prompt_boxes 的10%外扩规则。原 A/B 风险统计与锁文件属于另一研究，保留原功能，不移植其阈值为 IACS 阈值。

发现的主要缺口：无批量压缩接口；无缩放后的提示与掩膜坐标往返；逐例预测未保存；无 HD95/ASSD/Boundary F1；无自适应压缩决策；无患者聚类的压缩重复测量分析；无新研究的续跑数据库、工作台和导出流程。旧 Dice 对双空返回1，只能用于明示的预测一致性约定，不能泛化成无效人工轮廓得分。

## 6. 拟新增模块与修改范围

仅在扩展目录新增模块：数据清单与QC、SAM适配器、压缩、批量任务SQLite记录、指标、无真值特征、IACS配置、统计、导出、本地工作台、命令行、测试与文档。具体文件见 DEVELOPMENT_PLAN.md。旧脚本/配置先做只读哈希清单和小文件备份；不删除现有功能，不写原图和权重。

## 7. 风险与处理

1. 患者/病例定义未证实：保留来源字段和缺失状态；需要患者语义时拒绝推断，补充来源证据前仅输出病例级描述。
2. 外部独立性未证实：记录像素重复结果与来源证据，禁止以哈希不重复替代患者来源审查。
3. 高位深/DICOM：先识别；未实现可靠转换的编码显式拒绝，不暗转8位。现有PNG支持范围以实际检查为准。
4. IACS参数、Dice非劣界值和HD95容忍界值无本研究先验证据：0.02仅候选，不自动锁定为科学阈值；规则能执行不等于性能已验证。
5. 失败与空预测：保留任务和错误状态；边界不可定义时保存缺失与原因；统计报告完整计划分母、有效分母与缺失。
6. 隐私：所有医学像素本地处理；普通日志使用数据集内去标识标识，不上传外部服务；数据集内编号不当作通用个人身份。
7. 原始保护：新输出严格限定在独立 research_output；禁止输出覆盖输入路径，完成后核对数据与权重哈希。

## 8. 审计结论

存在可复用研究资产，应新增实验扩展而不是重建原工程。允许推进模块开发、真实开发集技术试跑与测试；正式 SCI 非劣效结论、独立外部验证和可靠 U-Net 跨模型结论仍需相应证据与预设方案。平台开发状态与科研验证状态必须分别记录。

## 9. 后续复核补充（不改写初始发现）

全量当前文件核验：1875对BUS-BRA与252对BrEaST均可读、主参考非空、图像轮廓尺寸一致，无精确像素重复；BrEaST原工作簿另4正常无主病灶，清单明确登记。BUS-BRA均8位L，BrEaST均8位RGBA。

BrEaST原论文核实256图来自256患者，CaseID与官方文件配对后记录patient_id_verified=True。实际工作簿256行Pixel_size均可读取，原文为cm，已换算为mm并保留原值/来源，不能从此前PNG内无spacing推断数据整体无间距。BUS-BRA Case列语义仍待直接来源定义。[BrEaST原论文](https://www.nature.com/articles/s41597-024-02984-z)、[BUS-BRA摘要](https://pubmed.ncbi.nlm.nih.gov/37937827/)。

隔离环境实际torch2.4.1+cu118/torchvision0.19.1+cu118正常，CUDA可用；新增依赖仅位于扩展venv。真实技术试跑已完成GPU63项、CPU5项，首轮确定性63项失败保留，尚无正式外部压缩实验。
