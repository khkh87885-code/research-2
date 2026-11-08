# SAM 压缩研究扩展交付

2026-11-08，首版 0.1.0。实际安装位置 `D:\Codex工作区\busbra_sam_pilot`。

先看 PROJECT_AUDIT.md、IMPLEMENTATION_PLAN.md，再按 USER_MANUAL.md 操作。新增代码位于 compression_study，已有六个科研脚本、旧方案、旧阈值与结果均保留。SAM 源码和权重不包含在交付压缩包内，继续使用本机已核验文件。

```powershell
Set-Location -LiteralPath 'D:\Codex工作区\busbra_sam_pilot'
& '.\scripts\Test-CompressionEnvironment.ps1'
& '.\scripts\Invoke-CompressionStudy.ps1' -Command test
```

已有技术验收的最终运行目录为 `research_output\smoke_20261108_v2`。本地结果查看入口为其中的 dashboard.html，表格和图表在 export 子目录。再次推理需新建配置和输出目录，或对完全一致的配置显式 --resume；手册给出完整命令。

科学边界：这是一套可执行的研究工具和小批量技术验收记录，尚不是完成的压缩 SCI 研究。正式开发/校准、预设界值、锁定 IACS、内部与跨数据集验证、样本量和应用定位方案仍需按 EXPERIMENT_PROTOCOL.md 完成。平台没有发布、联网推理或图像上传功能。

源码交付包仅包含新增代码、配置、测试、文档和依赖声明，不含医学图像、掩膜、权重、患者原始清单、匿名化密钥或虚拟环境。解压后应合并到已审计的现有项目，不用它重建或覆盖整个旧项目。

已知界面验收限制：浏览器工具的策略拒绝 file: 协议，因此未完成浏览器中的视觉与交互验收。HTML 静态检查、JavaScript 语法检查和数据导出测试已完成；PNG 汇总图已实际打开检查。用户可在本机浏览器自行打开 dashboard.html 查看。
