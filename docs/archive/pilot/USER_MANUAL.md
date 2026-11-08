# Windows 本地压缩分割实验平台使用说明

日期：2026-11-08。此平台以新增 `compression_study` 包扩展现有项目；旧推理和旧研究分析脚本仍保留。软件验收、少量真实开发病例试跑和正式研究必须分别报告。本文不声明 SCI 正式研究已完成。

## 目录与解释器

项目根：`D:\Codex工作区\busbra_sam_pilot`。所有以下命令在该目录的 Windows PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\Codex工作区\busbra_sam_pilot'
$studyPython = Join-Path $PWD '.venv-compression\Scripts\python.exe'
```

项目专用虚拟环境继承已核验的 PyTorch 2.4.1+cu118 / torchvision 0.19.1+cu118，新增科学计算依赖安装在项目环境中。不要为运行新平台重装旧环境或改写模型。当前系统 `PYTHONPATH` 会污染旧环境，命令统一加 `-E`；包装脚本已经处理，无需更改系统变量。

## 环境检查、备份和测试

```powershell
& '.\scripts\Test-CompressionEnvironment.ps1'
& '.\scripts\Backup-CompressionProject.ps1'
& '.\scripts\Invoke-CompressionStudy.ps1' -Command test
```

脚本执行策略限制时，可仅为这一次进程使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\scripts\Test-CompressionEnvironment.ps1'`；这不更改系统执行策略。包装运行和环境检查的 Transcript 写入 `research_output\logs`。备份只复制顶层 Python、Markdown、JSON 和配置文件，写入新的 `upgrade_backups\code_时间戳`，逐文件 SHA-256 核验，拒绝覆盖已有备份，不复制医学图像或权重。

## 全量清单核查

```powershell
& '.\scripts\Invoke-CompressionStudy.ps1' -Command inventory -Config 'configs/local.json'
```

读取现有冻结清单，不重新抽样划分。报告包括配对状态、实际解码尺寸/位深、文件哈希、完全重复字节组和失败数。输出目录自动使用时间戳；缺失或损坏记录不会被伪装成正常数据。HMAC 匿名化密钥保留在本机 `.compression_private\pseudonym.key`；不要发布密钥或原始清单。密钥丢失会改变匿名键，不能与旧运行拼接。

## 创建一次新的技术试跑

现成 `configs/smoke.json` 仅选 BUS-BRA 开发集中的少量病例，固定随机种子；内部留出集和 BrEaST 不参与这次开发试跑。新运行使用新配置和新输出目录：

```powershell
$runName = 'smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$runConfig = 'configs/' + $runName + '.json'
$studyConfig = Get-Content -LiteralPath 'configs/smoke.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$studyConfig.output = Join-Path $PWD ('research_output\' + $runName)
$studyConfig | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $runConfig -Encoding UTF8
$runDir = $studyConfig.output
& '.\scripts\Invoke-CompressionStudy.ps1' -Command run -Config $runConfig
```

运行保存代码/配置/输入/权重哈希，固定 SAM 与提示策略，逐任务写预测掩膜、参数、指标和事件日志。未修改的同一运行可以显式恢复：

```powershell
& '.\scripts\Invoke-CompressionStudy.ps1' -Command run -Config $runConfig -ExtraArguments @('--resume')
& '.\scripts\Invoke-CompressionStudy.ps1' -Command run -Config $runConfig -ExtraArguments @('--resume','--retry-failed')
```

已成功任务不重复推理；失败重试写新 attempt 目录，旧记录保留。恢复会核对代码、配置、输入文件和权重；任一变化需创建新运行。不要为绕过校验修改 `run.json`。CPU 运行需在新配置将 `model.device` 显式设为 `cpu`，耗时与 GPU 运行分开报告。

## 本地导出和结果工作台

```powershell
& '.\scripts\Invoke-CompressionStudy.ps1' -Command export -RunPath $runDir -OutputPath (Join-Path $runDir 'export')
& '.\scripts\Invoke-CompressionStudy.ps1' -Command dashboard -RunPath $runDir -OutputPath (Join-Path $runDir 'dashboard.html')
```

CSV/XLSX 包括所有图像任务、患者汇总、条件汇总和具备数据的统计检验表。另有统计 JSON、PNG/SVG/PDF 汇总图和真实状态说明。失败行仍保留；未定义值为空，不填成好成绩。导出保留匿名键、处理参数、IACS 实际选定条件、特征和指标，不带原始患者 ID、原图绝对路径或医学图像。

`dashboard.html` 可用本地浏览器打开，提供数据集、split、条件、提示、状态筛选和筛选后 CSV 下载。它是自包含的结果查看工作台，无外部库、无联网请求、无上传服务；图像均值图仅供浏览，统计推断应读取患者级导出结果。运行文件、任务记录和模型目录不能作为导出目标。

如需额外混合效应拟合，可为导出选择新的目标目录并加参数；模型可能因样本不足或不收敛返回不可解释状态，不能隐去警告：

```powershell
& $studyPython -E -m compression_study export --run $runDir --output (Join-Path $runDir 'export_mixed') --mixed-effects
```

## 开发集拟合 IACS 的入口

本次技术试跑不拟合正式策略。后续应先在新开发配置中明确保留一种 `prompt_strategies`，运行 `configs/development.json` 的完整开发条件；当前拟合接口拒绝同图同条件的重复行，因此不要直接输入包含两种提示的运行。机制与应用提示需各自开发、验证与锁定策略。

预先填写 `policy_settings.json` 的 `delta`（允许 Dice 损失）和 `hd95_delta`（允许 HD95 增量，当前单位 pixel），连同依据、特征和患者划分参数归档。下面结构中的 `null` 是“尚未指定”，会被程序拒绝；不是可用的默认研究阈值：

```json
{
  "delta": null,
  "hd95_delta": null,
  "feature_names": ["clarity", "noise", "contrast", "edge_density"],
  "feature_version": "full_image_v1",
  "conditions": ["baseline", "jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "jpeg_q40"],
  "seed": 20261108,
  "validation_fraction": 0.25,
  "max_depth": 3,
  "min_samples_leaf": 2
}
```

界值已经预先确定、开发集已实际运行并核验后，调用：

```powershell
# $developmentRun 指向仅一种提示的 BUS-BRA development 运行目录。
& $studyPython -E -m compression_study fit-policy --run $developmentRun --settings 'configs/policy_settings.json' --output 'research_output/locked_policy_v1.json'
```

策略文件需使用新文件名，随后在新的验证配置中通过 `adaptive.policy_file` 指向该锁文件。不要输入 internal_holdout/BrEaST 结果，不要用外部结果反复改界值或树参数，不要把开发标签准确率称作分割非劣。

## 当前方法和限制

- `mechanism_gt_box` 复用原参考掩膜扩框，先在原图确定框再映射；属于使用 GT 定位的机制实验。`application_full_frame` 使用整幅图框，不读取 GT 定位，但尚未经病灶定位验证。两组不可混合成一个应用效果结论。
- 缩放使用实际四舍五入后的尺寸映射框，预测掩膜最近邻还原到原图网格。还原尺寸不代表恢复丢失信息。
- 本版图像输入为明确的单帧 8 位图像。高位深、DICOM、多帧和非恒等 EXIF 方向输入会拒绝处理，需另行审计转换，不会静默转成 8 位。
- BrEaST alpha 按原项目黑底合成；PNG 的无损性是处理用 RGB 像素相等，不是原文件字节相等。灰度原文件和 RGB 派生容器的体积差异应如实解释。
- CR3 明确为编码后解码、缩小并以无损 PNG 保存最终负载；分别记录中间体积与最终负载体积。不得把中间 JPEG 体积冒作最终上传体积。
- 距离在没有可靠像素间距时使用 pixel，不是 mm。空预测的 Dice/IoU 为 0，而表面距离缺失；空参考判无效，不自动赋值为完美分割。
- 上传耗时按文件字节数及配置带宽模拟，不是实测网络延迟。编码、解码、推理时间各自记录。
- IACS 默认规则是未验证工程原型。`fit_policy` 为 BUS-BRA 开发病例提供患者隔离的树模型拟合接口，需预先指定 Dice/HD95 界值；外部/留出数据不可用于拟合。当前没有把这些默认阈值视为医学安全标准。
- 患者来源独立性、近重复、训练数据重叠、样本量及正式终点仍需研究者确认。小规模技术试跑的 P 值、区间和图不构成正式论文效能证据。

更完整研究闸门见 `EXPERIMENT_PROTOCOL.md`；实际执行结果及测试状态以本次验收报告和运行日志为准。
