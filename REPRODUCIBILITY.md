# Windows PowerShell 复现

## 1. 校验与数值重放

克隆后在仓库根目录运行。校验脚本只需 Python 标准库；不会打开患者图像或加载 SAM。

```powershell
git clone https://github.com/khkh87885-code/research-2.git
Set-Location -LiteralPath .\research-2
python -E .\scripts\verify_publication.py
```

原数值环境为 Python 3.8.20、numpy 1.24.4、scipy 1.10.1、scikit-learn 1.3.2。使用已有匹配环境或新建隔离环境后安装公开清单。以下安装命令是供读者执行的说明，本次上传没有修改原环境。

```powershell
py -3.8 -m venv .venv-analysis
$PythonPath = (Resolve-Path -LiteralPath '.\.venv-analysis\Scripts\python.exe').Path
& $PythonPath -E -m pip install -r .\pilot\requirements-iacs-v2.txt
& $PythonPath -E .\scripts\reproduce_numeric_analysis.py
# 可选：从冻结匿名数据重新拟合完整外5/内3 CV，输出目录必须为空/不存在。
& $PythonPath -E .\scripts\reproduce_numeric_analysis.py --refit-output .\local_analysis\refit_v1
```

脚本重放852条外层决策，并在故意破坏评价结局后验证选择不变。完整 refit 还比较 OOF 选择和全开发冻结策略哈希。数值库与原折分必须配套使用；私有密钥未公开，新生成的伪名可能改变 GroupKFold 排序。版本变化可能产生数值差异，应如实记录。

## 2. 查看科研输出

`reports/iacs_v2/research_reports/` 为本轮科学复核报告；`tables/IACS_v2_SCI_results.xlsx`（位于 experiments 对应运行下）为12页补充科研表。publication_v2 的11页展示工作簿是另一份较早展示导出，不能误称同一文件。

大文件使用 `.gz` 无损压缩；文件清单同时保存原字节哈希。要打开离线 HTML，可只解压到被忽略的本地目录：

```powershell
New-Item -ItemType Directory -Path .\local_analysis -Force | Out-Null
& $PythonPath -E -c "import gzip,pathlib; p=pathlib.Path('reports/iacs_v2/publication_v2/dashboard.html.gz'); pathlib.Path('local_analysis/dashboard.html').write_bytes(gzip.decompress(p.read_bytes()))"
Invoke-Item -LiteralPath .\local_analysis\dashboard.html
```

这些报告只展示数值和汇总图，不提供逐例临床像素浏览。当前旧工作台链接中本地端口是原机器入口，不是公网服务。

## 3. 运行现有源码测试

```powershell
Push-Location -LiteralPath .\pilot
& $PythonPath -E -m pytest .\tests .\tests_iacs_v2 -q
Pop-Location
Push-Location -LiteralPath .\workbench
& $PythonPath -E -m pytest .\tests -q
Pop-Location
```

完整旧工作台部分测试和 SAM 推理依赖原 torch/torchvision；原环境为 torch 2.4.1+cu118、torchvision 0.19.1+cu118。该依赖在原环境通过继承获得，公开数值分析不需要 GPU。不要将继承环境说明误解成仅凭 requirements 文件已安装 torch。

## 4. 自行重新采集图像指标

需要自行取得数据、正确分组清单、SAM 官方源码和 checkpoint。复制 `pilot/configs/*.example.json` 或 `workbench/config.example.json` 为私有配置，修改路径和新运行目录。原机器配置只作为 `docs/config_snapshots/` 与运行 configs 内的历史快照保存。不要更改旧冻结锁来绕过路径或哈希检查。

pilot 的 PowerShell 入口位于 `pilot/scripts/Invoke-IACS-v2.ps1`，支持 collect/prepare/validate/statistics/report/test/decide。先完成预检与 CPU/GPU 技术小样，再运行新版本采集；原图、权重只读。根目录各原始推理脚本保留于 pilot 以满足既有代码导入。旧工作台按其手册在 workbench 目录启动，配置需先自行复制并编辑。

本次采集与交叉验证的原运行日志、原文件保护检查、测试结果均已归档。读者重新运行是新实验，不能将其结果自动覆盖为本次冻结结果。正式非劣效界值和像素间距未确定，任何复现都不能据此新增临床结论。
