# 首版工程验收报告

日期：2026-11-08。最终源码版本 0.1.0，真实验收运行 `research_output/smoke_20261108_v2`。不将合成测试或少量技术试跑记为 SCI 正式实验。

## 测试结果

| 阶段 | 实际执行与结果 |
|---|---|
| 审计及备份 | 修改前备份 10 个原顶层文件，逐文件哈希相等；开发后 10/10 原文件哈希不变 |
| 环境 | 新 .venv-compression 复用 Python3.8.20/torch2.4.1+cu118；PowerShell 环境脚本通过，GPU 与四种编码器可用 |
| 数据清单 | 213 开发 + 851 内部留出 + 252 BrEaST；1316 对全部解码/尺寸/非空参考通过，失败 0；开发留出患者交叉 0 |
| 原流程回归 | 调用旧 run_formal_validation.preflight()，返回 preflight_verified，851+252；旧结果未重跑或覆写 |
| 最终单元及集成测试 | `python -E -m pytest tests -q`：41 passed in 10.55s；XML 与完整日志归档 |
| 真实 SAM | ViT-B 权重实际加载，3 位 BUS-BRA 开发患者 × 2 提示 × 17 条件 = 102/102 成功 |
| 恢复与失败 | 合成测试强制失败/重试/恢复、配置变化、损坏产物、输入路径越界及禁止覆盖通过；真实运行显式恢复通过且不增加预测任务 |
| 导出 | 逐图/患者/条件/配对/Friedman CSV、XLSX、统计 JSON、PNG/SVG/PDF 实际生成并可读取；混合效应接口实际调用，警告保留 |
| 工作台 | 本地自包含 HTML 已生成；Python 导出测试、JS 语法检查通过；浏览器视觉/交互验收未完成，详见下文 |
| 数据安全 | 最终运行前后逐图/掩膜、权重哈希不变；SAM Git clean；无医学图像上传 |

## 测试覆盖与日志

测试包括解析几何 Dice/IoU/边界距离、两方向距离及 spacing、空参考/空预测、PNG/WebP/JP2 无损往返、16 位（含 RGB16 PNG）拒绝、alpha 黑底、CR2/CR3 顺序、实际尺寸框映射、最近邻标签恢复、图像特征无 GT、患者内多图配对/失败分母、Bootstrap/Holm/Friedman、混合模型失败状态、患者隔离的策略拟合和外部数据拒绝、匿名导出与公式/HTML 转义、缺失患者身份拒绝及已有文件保护。

原始测试日志保留了开发期两项测试 fixture/断言错误及其修正，未删去失败记录。最终验收以 tests_release.xml、tests_release.log 为准。真实执行日志 real_sam_release.log、release_resume.log、release_export.log、release_dashboard.log，以及 PowerShell Transcript、环境依赖记录和 original_integrity_check.json 另存归档。

首次 Python3.8 直接联网安装因 TLS EOF 失败；随后由本机现代 Python 从官方包索引下载兼容 wheel，在新增虚拟环境离线安装成功。未重装或改动原 PyTorch 环境。完整依赖版本与 wheel SHA-256 已保留。

## 原功能保护

未编辑 6 个旧 Python 脚本、旧方案、旧统计 JSON 或旧阈值锁。旧源码的运行入口和阈值语义保持原状。本次核验旧 preflight 成功；不宣称重新推理验证了全部既有研究结果，也不覆盖旧 formal_validation。

## 尚未通过或尚未进行

- 浏览器工具拒绝打开本地 `file:` HTML（URL 协议策略），因此没有浏览器中真实点击筛选、下载按钮或视觉截图的验收证据。没有绕过该限制，报告不把静态检查写成界面通过。PNG 汇总图已实际打开检查；它是生成图表，不是桌面截图。
- 未实际执行 CPU 全流程推理；CPU 设备分支已实现，真实验收使用 RTX3060 Laptop GPU。
- 未进行正式压缩研究的全开发集、内部留出或 BrEaST 推理；只做全清单数据核验。未进行正式样本量、应用定位器、严格计时/真实网络或临床效用验证。
- 未训练 U-Net，未声称跨模型泛化；本机未发现可直接复用的已训练 U-Net。

工程首版可执行；正式 SCI 证据仍按 EXPERIMENT_PROTOCOL.md 的闸门推进。
