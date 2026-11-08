# IACS_v2 探索性报告

探索性 Case-level 分析；Dice 下降 0.02 仅为候选。HD95 容忍界值未设，不能判断双约束达标、临床安全或非劣效。

{
  "cases": 213,
  "compression_records": 1491,
  "valid_dice_records": 1491,
  "failed_or_missing_records": 0,
  "status_counts": {
    "ok": 1491
  },
  "outer_decisions": 852,
  "outer_cases": 213,
  "fallback_decisions": 10,
  "outer_folds": 5,
  "v2_condition_counts": {
    "jpeg_q95": 110,
    "jpeg_q75": 32,
    "jpeg_q85": 51,
    "jpeg_q60": 14,
    "baseline": 5,
    "jpeg_q40": 1
  },
  "v2_fallback_decisions": 5
}

失败和缺失保留在原始分母。训练描述、候选预测与外层观测不混用。所有图为本地聚合图，不含医学图像。

- Figure 1 研究流程（方法图，非实验结果）: method_only
- Figure 2 原图特征分布: observed
- Figure 3 JPEG 体积与分割质量: observed
- Figure 4 外层验证 IACS 与固定条件: observed
- Figure 5 原图特征与决策关联: observed
- Figure 6 外层交叉验证各折: observed
- Figure 7 训练折阈值定义的亚组: observed
- Figure 8 逐 Case 体积 / Dice / HD95 Pareto: observed