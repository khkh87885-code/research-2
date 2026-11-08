# IACS_v2 Windows 使用说明

本轮仅 BUS-BRA 冻结开发清单213 Case。统计单位为Case，不将Case自动等同独立患者。输出根为 `research_output/IACS_v2/dev_20261108_123935`；旧 compression_study、旧41项测试和旧102任务均保留。

## 查看本次交付

- 先读 `reports/research_reports/IACS_SCI_RESULTS.md`，再读模型卡和统计报告。
- 五页只读工作台：<http://127.0.0.1:8766/iacs-v2>；原八页工作台仍为8765。
- 逐图数据：features/iacs_features.csv；metrics/Compression_Sensitivity_Dataset.csv；metrics/compression_sensitivity_long.csv。
- 外层OOF：cross_validation/oof_decisions.csv、candidate_predictions.csv、validation.json。
- 最终统计：statistics/review_v2/statistics.json；风险描述：statistics/risk_evaluation.json / CSV。
- 逐图中文解释数据：tables/decisions_explained.csv；图/Excel/脱敏报告：reports/publication_v2。
- 完整真实预测与压缩版本保留在compression/tasks；OOF算法的选择引用这些同图同模型预测，不重复伪造掩膜。

当前结果是质量风险与文件体积的权衡，没有证明IACS相对固定Q85同时更省空间且更可靠；正式HD95容忍界值及临床非劣效界值仍未设。

## PowerShell 入口

```powershell
Set-Location -LiteralPath 'D:\Codex工作区\busbra_sam_pilot'
& '.\scripts\Invoke-IACS-v2.ps1' -Step test
# 对已冻结、完全相同的 collection 恢复；不会重复已成功推理。
& '.\scripts\Invoke-IACS-v2.ps1' -Step collect -Resume
```

运行命令统一用 `.venv-compression\Scripts\python.exe -E`，规避全局PYTHONPATH污染；可通过-PythonPath指定已经验证且依赖相同的Conda解释器。没有升级Python/PyTorch/CUDA/SAM，新增需求列在requirements-iacs-v2.txt，本轮无需新增安装。

## 创建后续新版本

复制配置并设置新的run_root，保持datasets只含BUS-BRA development及原manifest SHA-256；不要更改原清单。collect每条任务有独立attempt，重试使用 `-Resume -RetryFailed`，保存失败记录。正式研究用途、样本或策略改变应新建版本，不修改本轮冻结配置/JSON/掩膜。

```powershell
$newConfig = 'configs/iacs_v2_your_new_version.json'
& '.\scripts\Invoke-IACS-v2.ps1' -Step collect -Config $newConfig
& '.\scripts\Invoke-IACS-v2.ps1' -Step prepare -Config $newConfig
& '.\scripts\Invoke-IACS-v2.ps1' -Step validate -Config $newConfig
```

prepare/validate均拒绝覆盖已有阶段输出。复用旧结果会检查源/权重/代码/提示和实际recipe，不能只凭Q值或文件名复用另一平台的灰度编码结果。PNG仅在实际RGB逐像素相同且同提示时复用baseline预测，耗时标明reuse；不可把0毫秒当新模型速度。

## 单张原图决策（不读人工轮廓）

```powershell
$run = 'research_output/IACS_v2/dev_20261108_123935'
$image = '用户指定的原图完整路径.png'
& '.\scripts\Invoke-IACS-v2.ps1' -Step decide `
  -ImagePath $image `
  -PolicyPath "$run/cross_validation/full_development_policy.json" `
  -OutputPath "$run/deployment/new_image_decision"
```

该入口只读原图，提取26特征并测得全部候选编码体积，保存逐候选预测/理由；不接受mask、SAM预测或真实Dice作为输入。返回选择不等于分割质量已达标。输入仅支持已审计的单帧8位图像；高位深、方向或DICOM转换仍需另行审计。

## 重现论文报告

```powershell
& '.\.venv-compression\Scripts\python.exe' -E `
  '.\analysis_scripts\Reproduce-IACS-v2-Reports.py' `
  --config '.\configs\iacs_v2_development.json'
```

该脚本绑定本次已审阅数据的SHA-256，重新生成一个独立reports/reproduced_时间戳目录；改变数据会明确拒绝，防止旧数值叙述被套到新结果。下一版研究须重新审阅叙述。一般数值验证与图表输出由iacs_v2模块提供，不能自动把软件测试通过写成算法有效。

## 只读页面重启

若8766已正常响应，不重复启动；关闭后可执行：

```powershell
& '.\.venv-compression\Scripts\python.exe' -E -m iacs_v2.viewer `
  --config configs/iacs_v2_development.json --report publication_v2 --port 8766
```

页面仅监听127.0.0.1，固定清单文件及哈希，拒绝写入接口，不开放原图或模型目录。原服务重启曾被自动审批拒绝，未给详细理由；本轮保留它，专项页面使用独立只读服务。原工作台已增加通往8766的入口。当前server.py中受限报告路由也已实现并通过测试，原运行进程不会热加载这些Python路由，不声称它已被重启。

## 统计边界

Dice下降0.02只作用户允许的候选；HD95未设容忍界值，因此没有联合约束达标或非劣效判定。HD95/ASSD为pixel，缺可靠逐图spacing不换算mm。2000次Case bootstrap条件于固定OOF记录，不包含重拟合/选型或折间训练依赖；风险AUC另用500次Case重抽样。模型可执行不等于概率已校准。不得用原851留出集或BrEaST反复优化本轮策略。
