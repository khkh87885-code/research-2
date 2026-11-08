# SAM 本地项目审计

审计日期：2026-11-08（Asia/Shanghai）。方法：只读检查源代码、目录、清单及本地环境；开发前完成。未上传医学图像，未改原图或权重。

## 结论

实际项目根为 `D:\Codex工作区\busbra_sam_pilot`；SAM 源码为同级 `segment-anything`；数据根为 `D:\codex论文`。当前聊天目录不是科研项目根。应采用新增模块扩展，保留所有旧脚本、锁文件、研究方案和结果，不重建或搬迁项目。

现有研究主题是分割失效预警阈值跨数据集验证，不是图像压缩。已有正式结果只能作为旧研究证据，不能充作新增压缩实验结果。

## 架构与运行入口

纯 Python 批处理，无已接入前后端、数据库或文件上传服务。SAM 上游 demo 独立存在。项目自身无 Git 仓库；SAM Git 干净，HEAD 为 `dca509fe793f601edb92606367a655c15ac00fdf`。未发现适用 AGENTS.md。

| 文件 | 已有能力与复用方式 |
|---|---|
| run_single_case.py | 单例推理、GT 扩框、四向扰动、比较图；34–59 行 prompt_boxes 可直接复用 |
| run_development_pilot.py | 40 例先导；40–85 行推理流程可包装复用 |
| run_full_development.py | 213 例开发、失败重试、阈值分析 |
| run_formal_validation.py | BUS-BRA/BrEaST 清单和锁核查、正式评分；74–77 行 rgb_over_black 可复用 |
| analyze_formal_validation.py / analyze_validation_supplementary.py | 旧失效预警研究统计，不更改其含义 |
| formal_validation_threshold_lock.json | 已锁定旧研究阈值，不修改、不用于压缩算法调参 |

原运行方式：用 learn_pytorch Python 执行各脚本；`run_formal_validation.py` 默认只核查，`--score` 才推理且拒绝覆盖已有正式结果。旧代码绑定 CUDA/特定版本和硬编码路径，普通运行可能受全局 PYTHONPATH 污染。新增 PowerShell 入口只在当前进程清除污染并复用原环境，不修改系统环境或重装 PyTorch。

## 数据和模型

| 对象 | 本次已核验状态 |
|---|---|
| BUS-BRA 开发清单 | 213 行，213 患者/图像，全部图像和掩膜存在 |
| BUS-BRA 内部留出清单 | 851 行，851 患者/图像；与开发患者交叉 0 |
| BrEaST 已纳入清单 | 252 行，252 患者/图像，配对文件存在 |
| BUS-BRA 数据 | D:\codex论文\BUSBRA；原包及既存划分位于数据根 |
| BrEaST 数据 | 项目 breast_external_extracted；清单及锁位于 breast_external_test |
| SAM | 官方 SAM v1 ViT-B，非 SAM2；单掩膜推理 |
| 权重 | D:\codex论文\models\sam_vit_b_01ec64.pth；375042383 字节 |
| 权重 SHA-256 | ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912，与旧代码一致 |

清单已有 dataset、split、patient_id、image_id、pathology、image_path、mask_path、width、height，BrEaST 另有源文件哈希。现有流程每患者一张图，不能据此宣称两库所有图像均已纳入新研究。患者来源独立性及模型训练数据重叠仍不能仅靠文件去重证明；对外称跨数据集验证，独立性待来源证据确认。像素间距缺失时边界距离只能用像素，不冒用 mm。

## 推理与坐标

原图 RGB；BrEaST 透明通道按黑底合成。参考掩膜非空。人工参考框每侧扩张 10%，另有四向 5% 扰动。这属于 GT 定位的机制实验；新增应用策略必须不用 GT。固定框先在原图生成，再按实际处理尺寸映射。SAM 返回当前输入尺寸掩膜，缩放实验须用最近邻还原到原图后评价。

现有 Dice 将双空记 1，只保留为旧研究一致性约定。新评价对双空、空参考、空预测分别记录状态，边界不可估计值保留缺失，不能替换成好成绩。

## 环境

Windows PowerShell，RTX 3060 Laptop 6144 MiB，驱动 617.42。可复用解释器 `D:\Miniconda3\envs\learn_pytorch\python.exe`，PyTorch 2.4.1+cu118，CUDA 可用。base Python 3.14 无科研依赖；env_ml Python 3.6 无 torch，不使用。

发现全局 PYTHONPATH 指向 base site-packages，污染 learn_pytorch 导入；隔离调用 `python -I` / `-E` 验证 torch 正常。新入口使用 `-E` 忽略外部 Python 环境变量并保留脚本路径，完整依赖证据附 environment_audit.json。审计本身不改环境。

Python 3.8.20；NumPy 1.24.4、Pillow 10.4.0、matplotlib 3.7.5、torchvision 0.19.1+cu118 已有。缺 SciPy/pandas/statsmodels/openpyxl/pytest/scikit-learn。Pillow JPEG/PNG/WebP/JPEG2000 均可用。后续新增项目专用虚拟环境，继承原 torch，只在新环境补充科学计算依赖，保护原环境。

数据审计补充：BUS-BRA 实际 1875 对图/掩膜、1064 官方 Case；原图 8 位灰度。冻结清单的 213+851+252 对均解码且尺寸一致。两库文件字节哈希内外完全重复为 0；近重复未查，患者来源独立性不能由此证明。BrEaST 8 位 RGBA，必须复用黑底合成，历史锁记载 8 例忽略 alpha 会改变 RGB。无可靠 Lesion_ID 或 spacing，不造编号/单位。

## 缺口、风险与控制

- 缺批量压缩、缩放、HD95/ASSD、新压缩研究统计、断点任务和依赖清单：新增 compression_study 包、configs、scripts、tests。
- 项目无 Git：先备份原脚本/方案/锁及哈希，不复制患者图像和大权重；开发后核验原文件哈希。
- 高位深/DICOM 不可静默转 8 位：首版拒绝未声明的转换，保留原文件，输出明确错误。
- 原结果、原图、权重不可覆写：新增结果独立运行目录；拒绝输出进入数据目录、权重目录或复用完成运行。
- 新 IACS 不能使用外部真值调参；先开发集技术测试，锁定策略，正式阈值与非劣界值待研究者预先指定。
- 原研究有已完成正式输出：不得重新调用 --score 覆写，也不把旧阈值称为压缩安全阈值。
- 普通日志仅匿名键/聚合计数/错误类型；原始路径映射保留本机私有清单，不进入论文导出。

## 预计新增文件

`compression_study/{datasets,model,engine,imaging,metrics,features,adaptive,statistics,export,cli}.py`、`configs/compression_*.json`、`scripts/*Compression*.ps1`、`tests/`，以及本审计、实施计划、实验方案、使用说明和测试/结果报告。六个旧脚本和旧锁文件不编辑。

本报告区分静态发现、环境导入验证和后续真实推理；模型实际加载/压缩推理在开发后的技术验收中另记，不冒称审计已完成全部压缩实验。
