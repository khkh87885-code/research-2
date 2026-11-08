# IACS v2 交付入口

2026-11-08。仅 BUS-BRA development，Case-level，探索性分析。Dice 损失0.02仅为用户允许的候选；HD95容忍界值未设，不作临床非劣效或安全声明。

完整开发集探索性验证已完成，尚未证明相对固定Q85的体积优势。先阅读 IACS_SCI_RESULTS.md，再看 VALIDATION / STATISTICAL / MODEL_CARD。

实际结果目录：`D:\Codex工作区\busbra_sam_pilot\research_output\IACS_v2\dev_20261108_123935`。原始矩阵在compression/tasks，敏感性库在metrics，26特征在features，OOF与策略在cross_validation，Case统计在statistics/review_v2，风险描述在statistics/risk_evaluation.json，八图及Excel在reports/publication_v2。

只读专项工作台：[本机 IACS v2](http://127.0.0.1:8766/iacs-v2)。原服务8765保持运行；重启被自动审批拒绝，工具未给详细原因，因此新页面使用独立只读服务，无实验启动或原图接口。

所有数值来自真实本地结果。未上传医学图像，未更改权重、原始清单或旧实验。
