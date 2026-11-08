# IACS v2 嵌套交叉验证

2026-11-08。仅 BUS-BRA development，Case-level，探索性分析。Dice 损失0.02仅为用户允许的候选；HD95容忍界值未设，不作临床非劣效或安全声明。

固定外层5折、内层3折GroupKFold，Group为现有Case。三个低复杂度家族为浅规则树、逐质量Logistic风险分类、逐质量Ridge变化回归。标准化、常量特征删除、风险参数、复杂度和最优固定质量均仅用对应训练折。外层结局不选冠军。

| 外层折 | 训练Case | 验证Case | 内层选择 | 风险参数 | 训练内最佳固定 |
|---|---:|---:|---|---:|---|
| 0 | 170 | 43 | B_logistic | 0.1 | jpeg_q85 |
| 1 | 170 | 43 | B_logistic | 0.1 | jpeg_q85 |
| 2 | 170 | 43 | B_logistic | 0.1 | jpeg_q85 |
| 3 | 171 | 42 | B_logistic | 0.1 | jpeg_q85 |
| 4 | 171 | 42 | B_logistic | 0.1 | jpeg_q85 |

独立复核：15个fold×family训练标准化与训练数据重新计算一致；852条OOF决策用保存policy重放完全一致；将验证候选Dice/HD95/ASSD/status改成极端错误值后，852次决策不变，说明部署决策不读取评价结局。各Case只出现在一个外层验证折。

IACS_v2 OOF质量分布：{"jpeg_q95": 110, "jpeg_q75": 32, "jpeg_q85": 51, "jpeg_q60": 14, "baseline": 5, "jpeg_q40": 1}，其中5例回退原图。五个外层训练内都选择B_logistic，训练内最佳固定都为Q85。不同质量分布是实际模型输出，不是验收时强制分配。全开发集最终策略用完整开发数据内层CV后重新拟合，其自身拟合表现不作为外层验证证据；尚无外部验证。

[GroupKFold官方说明](https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.GroupKFold.html)与[嵌套验证官方示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。
