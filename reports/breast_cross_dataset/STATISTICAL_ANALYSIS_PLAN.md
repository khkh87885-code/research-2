# BrEaST 冻结跨数据集验证：预设统计分析计划

版本：SAP v1；日期：2026-11-08（Asia/Shanghai）。在统计开发者读取本 run 的任何 BrEaST 新分割或效率结局之前写入并计算 SHA-256。方案不依据 BrEaST 结局选择模型、风险参数、质量规则、阈值或检验方法；修改方案必须另存版本并披露，不能覆盖本文件。

## 设计、分析单位和人群

固定三个配对组：Original、Q85、IACS。每个 Case 三组结果属于一个重抽样簇，不能作为三个独立样本。官方映射预期每 Case 一幅图；重复同 Case/图像/组或一 Case 多图作为输入冲突报告，不静默去重或改权重。

主要准确性人群：252 个有官方主病灶参考的 Case。4 个正常 Case 无主病灶参考，仅在 normal_transport 独立传输效率表中报告；另给 256 个 Case 的 all_transport 效率汇总，不把正常图像混入病灶分割指标。14 个其他轮廓不作为新的主病灶或独立观察。来源独立性尚不充分，结果表述为 cross-dataset / external dataset validation；不宣称已经确认独立多中心队列。

计划清单应通过 `planned_records` 输入统计程序。未产生结果的计划 Case 仍纳入计划分母与失败/缺失计数。不完整结果可生成明确标记的中间统计，但不得据此宣布整体验证完成。若未提供清单，只能从结果推断已见身份，同时显式列出预期 252/4/256 与实际已见数量之差，不虚构缺失 Case 的 ID。

## 终点和效应方向

四个预设主要终点：Dice、HD95_mm、ASSD_mm、saving_pct。HD95_pixel 与 ASSD_pixel 为单位附表。毫米数据仅来自已核验的真实 spacing 计算，不在统计阶段补造或统一填充 spacing。

次要指标仅作描述性汇总：IoU、precision、recall、boundary_f1、mask_consistency_dice、area_change_pct、encode_ms、decode_ms、inference_ms、output_bytes、original_bytes。normal_transport 和 all_transport 只汇总 saving_pct、encode_ms、decode_ms、output_bytes、original_bytes，不给无 SAM 推理的正常图像补零推理时间。

三项固定比较，均按前者减后者：IACS − Original、IACS − Q85、Q85 − Original。Dice 正差表示更高重叠；HD95/ASSD 负差表示更小距离；saving_pct 正差表示更多体积节省。均值差为配对 Case 差值的均值；中位差明确为配对 Case 差值的中位数，另附匹配组中位数之差以避免概念混淆。没有正式 Dice 非劣界值和 HD95 容忍界值，不做 NI 通过/不通过结论。

## 描述统计和共享配对 Bootstrap

连续终点报告计划数、收到记录数、成功处理数、有效数、缺失/失败数、均值、样本标准差（ddof=1）、中位数、Q1、Q3、均值和中位数的 95% CI。样本不足时 SD 或 CI 留空并说明，不填 0。分位数使用 NumPy 的线性插值约定。分类状态报告频数和分母/比例。

固定随机种子 20261108，5000 次 Case bootstrap，95% 百分位区间（2.5%、97.5%）。先按匿名 Case 键稳定排序，对每个人群一次性生成 5000 × N 的 Case 索引矩阵；同一 Case 的三组及所有终点均使用相同索引抽取。主分析的各组、三项配对比较、敏感性计数共享主病灶人群的同一抽样矩阵，保留重复抽到的 Case 的多重性。

有缺失时，在每次共同抽取的 Case 集中按相应终点的有限、有效测量计算组统计或成对差值。差值只使用两组同 Case 同时可评价者；不各自独立抽样、不用不配对的组均值替代配对均值差。抽样后无任何有效值的重复保留计数，不能赋值为 0；CI 使用有效重复且报告其数量。原始有效 Case 少于 2 时不报告经验 bootstrap CI。该 available-case 方案不消除非随机缺失偏倚，需同时展示完整分母和失败模式；不得将其写成自动完成了缺失数据校正。

## 配对检验、Holm 家族与三组支持检验

主要配对检验固定为双侧 Wilcoxon signed-rank，`zero_method='wilcox'`、无 continuity correction、SciPy `method='auto'`。检验对象是配对差值分布关于零的对称性，不应把其 P 值误解为均值或中位数的无条件检验。保存正/负/零差数量、差值偏度与软件警告；观察图形后不自动切换检验以追求显著。效应量和 bootstrap 区间是主要结果表达。

主要 Holm 家族预先固定为 3 比较 × 4 主要终点 = 12 项双侧检验，alpha=0.05。不可评价项的 P 保持缺失，校正仍按预定 12 项家族计算，不因缺失缩小家族。没有有效配对或不足 2 个有效 Case 时 P 为缺失。所有有效配对差均严格为 0 时，显式约定 statistic=0、P=1，避免库函数的全零报错；这不是非劣成立。Original saving_pct 全为定义性的 0 不属于错误，不删除该组；不同条件均为同一 Case 相同值时另标全组并列。

四个主要终点分别附三组完整 Case 的 Friedman 检验；不同终点分别求三组均有效的完整 Case，不跨组独立剔除。少于 3 个完整 Case 不计算；每个完整 Case 内三组均并列时 statistic=0、P=1。常规 Friedman 的 chi-square 近似在本设计仅 3 条件时存在局限，作为支持性结果并显式警告；四个 omnibus P 的 Holm 校正属于单独支持性家族，不占用或改变上述 12 项主要家族。像素距离附表和次要终点不新增未预设主要检验；配对像素差仅给效应和 CI。

## Dice 下降 0.02 的候选敏感性分析

分别对 Q85 − Original 和 IACS − Original 计算每 Case Dice 变化，报告下降严格超过 0.02 的数量、有效配对比例、95% Wilson 二项区间及同一 Case bootstrap 的比例 CI。阈值等值边界采用绝对误差 1e-12 的浮点容差，仅避免数学上恰为 0.02 的机器表示被误计为超过；不更改任何算法决策阈值。

明确列出计划 Case、可评价配对、超过候选值、没有超过候选值及未知/失败的数目。另给将全部未知分别视作未超过/超过时的计划分母比例范围，此范围不是 CI。正常 Case 无此分析。所有表注明候选、探索性、非临床界值；HD95 不设人为容忍值，报告实际差异和单位，不给双约束安全或 NI 标签。

## 失败和测量有效性

空预测在非空有效参考下，Dice=0 保留为有效重叠观测；HD95/ASSD 不可定义时为缺失，不从其他成功图像填充。空/无效参考不产生虚假的常规病灶 Dice。失败、未运行、无效参考、空预测表面缺失、spacing 缺失/不可靠、非有限值和其他未说明缺失分别统计；无法判别的原因保持 unspecified，不从一般异常类型猜测具体阶段。

分割指标要求成功推理或明确的有效空预测状态。已实测得到的编码/解码/文件体积效率即使后续 SAM 失败仍可纳入相应效率描述，失败任务仍保留在原分母。统计代码不改写任何输入行、不执行模型或优化、不从 BrEaST 重新标准化特征。

## 输出和软件

分析接口：`analyze(rows, output_dir, planned_records=None)`；只接收匿名键和数值/状态，不读取像素。输出新目录，拒绝覆盖已有分析；保存 `statistics.json`、描述/主要配对/像素附表/Friedman/候选敏感性/缺失分母 CSV 和方法简述。XLSX、图像和最终研究报告由独立导出模块使用这些真实结果生成。

使用现有 Python 3.8 / NumPy 1.24.4 / SciPy 1.10.1 环境，不为本分析升级原模型环境。模型、策略、标准化、提示和数据冻结由总流程核验；本计划不自行解冻或调整它们。结果必须把算法冻结完成与研究统计结论分开。

## 方法资料

配对重抽样采用相同索引的含义见 [SciPy 1.10.1 bootstrap](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.bootstrap.html)。Wilcoxon 的配对差值、零差处理与对称性解释见 [SciPy 1.10.1 Wilcoxon](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.wilcoxon.html)。Friedman 的完整配对数组、并列校正和近似局限见 [SciPy 1.10.1 Friedman](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.friedmanchisquare.html)。