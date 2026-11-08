# BrEaST 推理前冻结核验

状态：全部既有锁与依赖核验通过，尚未进行 BrEaST 推理或读取其验证结果。

冻结的是 213 个 BUS-BRA development Case 训练的完整开发策略 `full_development_policy.json`，不是任一外层折策略。模型为 B_logistic，risk_threshold=0.1；26项特征顺序、训练内scaler与所有模型参数逐字节/内容哈希固定。

- IACS v2 发布源码和原 compression/SAM/提示/图像/指标依赖均核对；SAM Git clean。
- 候选为原图、JPEG95/85/75/60/40、PNG level6，全部scale=1；JPEG subsampling=0。
- GT原图框每侧外扩10%，只生成一次；缩放按实际尺寸，掩膜最近邻恢复。
- Dice损失0.02仅候选；HD95容忍界值未设，边界单位px，不作临床非劣效。
- 特征提取与冻结策略决策不使用外部GT结局；机制提示与评价仍需参考轮廓，不称无真值自动定位。
- 未训练、未校准、未重选阈值，未改源文件/权重，未读取BrEaST影像或结果，未上传网络。

策略内容根哈希：`cfb6d4d536729fba888772818706f5ccb7b5785674f85e57795fb2a6a468b113`

Manifest内容根哈希：`d2e066d282e1b691492896a2c261cbde77e2d31f58a90142eb96ef4cc8a18d4a`

完整路径、逐文件SHA-256、环境版本、模型/提示/编码/评价定义记录于 `manifests/BREAST_FREEZE_MANIFEST.json`。后续推理和恢复前调用 `verify_freeze(runroot)`；任何改变立即阻断，不能回调参数或悄然改写冻结文件。