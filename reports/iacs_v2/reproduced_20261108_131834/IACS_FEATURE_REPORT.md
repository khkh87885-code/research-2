# IACS v2 原图特征报告

2026-11-08。仅 BUS-BRA development，Case-level，探索性分析。Dice 损失0.02仅为用户允许的候选；HD95容忍界值未设，不作临床非劣效或安全声明。

特征版本 `full_image_iacs_v2_1`，213图/213 Case、26项数值特征，缺失0。所有特征仅使用原图RGB；人工参考轮廓不进入特征函数。GLCM采用32级、距离1和2、4方向对称归一矩阵，先各方向求指标再等权平均。

| 特征 | 方法 |
|---|---|
| laplacian_variance | Variance of 4-neighbor discrete Laplacian; reflected boundary; gray in [0,1]. |
| tenengrad | Mean squared Sobel gradient magnitude; each Sobel derivative divided by 8. |
| gradient_mean | Mean normalized Sobel gradient magnitude. |
| gradient_p90 | 90th percentile of normalized Sobel gradient magnitude. |
| gradient_direction_entropy | 8-bin [0,2pi) direction entropy in bits, weighted by nonzero gradient magnitude. |
| noise_mad | Laplacian MAD / (0.6744897501960817 * sqrt(20)); uncalibrated robust noise proxy. |
| residual_mad | MAD of gray minus 3x3 reflected median filter / 0.6744897501960817. |
| high_frequency_ratio | Variance of gray minus sigma=1 reflected Gaussian / gray variance; constant=0. |
| global_contrast | Population standard deviation of normalized grayscale. |
| local_contrast | Mean local standard deviation in reflected 9x9 windows. |
| p01 | 1st percentile of normalized grayscale. |
| p99 | 99th percentile of normalized grayscale. |
| intensity_range | p99 minus p01 of normalized grayscale. |
| dynamic_range | Maximum minus minimum normalized grayscale. |
| gray_mean | Mean normalized grayscale; ITU-R BT.601 weights 0.299,0.587,0.114. |
| gray_std | Population standard deviation of normalized grayscale. |
| gray_skew | Third standardized central moment; a constant image has skew=0 by explicit convention. |
| edge_density | Fraction of pixels with normalized Sobel gradient magnitude > 0.05. |
| edge_direction_change | Mean circular angular change / pi between horizontal/vertical neighboring edge pixels; no pairs=0. |
| glcm_contrast | 32-bin symmetric GLCM sum(P*(i-j)^2); mean over valid distance1/2, four directions. |
| glcm_homogeneity | 32-bin symmetric GLCM sum(P/(1+(i-j)^2)); mean over valid offsets; constant=1. |
| glcm_entropy | 32-bin symmetric GLCM Shannon entropy in bits; mean over valid offsets. |
| histogram_entropy | 256-bin normalized gray Shannon entropy in bits. |
| width | Original decoded image width in pixels. |
| height | Original decoded image height in pixels. |
| aspect_ratio | Original width divided by original height. |

特征分布保存在diagnostics/diagnostic_payload.json，完整逐图表在features/iacs_features.csv。训练标准化仅使用训练折；常量特征仅在训练折判定并删除，不用全开发集结局选择特征。图中用于显示的全局z分数不参与CV。噪声、清晰度和形状均为工程描述量，不是物理噪声标定或临床质量量表。

260项特征-结局Spearman比较在Case内汇总，统一Holm校正后未发现P<0.05的关联；这不是证明特征没有作用。连续变化预测与风险分类的外层结果分别报告，禁止凭单一相关性指定风险权重。
