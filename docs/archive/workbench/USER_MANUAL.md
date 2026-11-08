# 使用说明

当前交付：本地科研实验扩展 v0.1，已完成真实技术试跑。正式科研验证尚未完成。

## 启动与打开

工作台仅监听本机：<http://127.0.0.1:8765>。本次交付时服务已启动，浏览器页已保留。重启电脑后，在PowerShell运行：

```powershell
& 'D:\codex学术论文2\启动科研工作台.ps1'
```

或直接运行：

```powershell
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' serve --port 8765
```

若端口已占用，先访问现有服务；不要重复启动。前台启动时Ctrl+C停止。当前后台进程编号在 `research_output/logs/server.pid`，停止前核实该进程仍为本项目服务。可用`serve --port 8766`另开本地端口。

## 路径和已有工程

config.json保存已确认路径：原数据和SAM权重位于 `D:\codex论文`，旧推理脚本位于 `D:\Codex工作区\busbra_sam_pilot`，官方SAM源码位于 `D:\Codex工作区\segment-anything`。扩展代码及所有新输出位于本目录。修改路径前备份配置；已有实验使用其冻结配置，不跟随全局配置变化。

`backups/20261108_original`包含15份旧脚本/配置/清单副本和SHA256记录。原始医学图像、权重、投稿材料、历史实验结果均未移动、删除或覆盖。

## 八页工作台

1. 项目总览：数据状态、来源内编号、真实任务进度、失败及运行设备。编号总和不是已证实的独立患者总数。
2. 数据集管理：读取两库全量清单、配对、尺寸/位深及质量状态；“原图/标注”只从本地接口显示去元数据PNG。4例BrEaST正常图无主轮廓有明确记录。
3. 压缩实验：选择开发集、图像数量、提示策略和设备，勾选处理组，“创建并运行实验”执行真实压缩和推理。数量0表示冻结选图清单的全部图像，不表示全1875图。默认每Case一图。
4. SAM分割：查看进度、错误、预测；续跑只执行未完成/失败任务，保留旧attempt。代码或配置变化会拒绝混跑，需新建实验。
5. 自适应压缩：从原图提取特征、生成候选配置、查询风险与压缩决策。当前候选已校准，重复校准拒绝覆盖。锁定按钮会因缺乏验证证据和质量界值而返回明确错误，这是当前预期行为。
6. 指标评价：查看原坐标下分割和压缩指标、真实文件体积-Dice散点。空值和原因保留。
7. 统计分析：当前BUS-BRA的Case患者语义未完成来源核验，返回描述统计并阻断患者推断。不能在看到结果后输入Δ来生成正式非劣效结论。
8. 论文图表与导出：每次生成独立CSV、Excel、PNG/SVG/PDF与Markdown报告。外部/亚组/跨模型缺数据的表明确为待完成，不填虚构数值。

## 两种提示实验

机制实验复用原图人工框，只评估给定定位条件下压缩的影响。应用实验用固定中央80%视野框，不读人工轮廓决定提示；该规则尚不是可靠病灶定位器。应用试跑首图Dice较低，真实保留，不把它作为临床自动分割系统推荐。

## 命令行复现

```powershell
# 完整扫描并建立数据QC；原图只读
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' scan

# 新建小规模机制试跑，自动生成新Experiment_ID
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' run --limit 2 --device cuda --prompt mechanism --variants baseline,jpeg_q85,scale_50,cr2_q75_s50,png

# CPU应用提示；与机制实验分开保存
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' run --limit 1 --device cpu --prompt application --variants baseline,jpeg_q85

# 已完成的真实GPU实验续跑将直接返回，不改写结果
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' resume EXP_574059083cd74bbc9e57b8fc52988266

# 导出新的独立快照
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' export EXP_574059083cd74bbc9e57b8fc52988266

# 全部自动测试 / 原文件和真实输出验收
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\run.py' test
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I 'D:\codex学术论文2\scripts\verify_delivery.py'
```

`capabilities`查看编码器能力。本机JPEG/PNG/WebP/JPEG2000均可用。CLI支持JPEG Q95/85/75/60/40、scale_100/90/75/50/25和_bicubic变体、cr2_q75_s50等组合、cr3_q75_s50（最终PNG）、cr5_s50、webp_lossless/webp_q85、jp2_lossless/jp2_lossy、iacs。完整配方见compression.py的recipes()。未支持的名称或编码明确报错。

## 输出与追溯

```text
research_output/
  dataset_report/inventory.json
  experiment_configs/iacs_candidate.json
  experiments/EXP_<uuid>/
    config.json, config_lock.json, tasks.sqlite3
    results.jsonl, summary.json
    artifacts/<image_id>/<variant>/attempt_<n>/
    model_session_*.json, integrity_*.json
    analyses/<uuid>.json
    exports/<timestamp_uuid>/
  figures/workbench_export.jpg
  figures/observed_panels/
  logs/automated_tests_final.txt, real_verification.json
```

完整逐例记录在SQLite和results.jsonl，模型会话记录源码提交、权重哈希、版本、设备与确定性限制。完成记录不可覆盖；代码变更须新建实验。原始图像/标注每次推理前后校验，所有处理副本保存独立目录。数据库与清单含本地路径和来源内去标识编号，分享前仍应按所在机构规范审查，工作台不向外部服务发送图像。

## 环境与常见问题

当前`.venv`继承已验证的learn_pytorch环境，额外依赖仅安装到本扩展环境；原Conda环境和权重未升级。使用`-I`隔离宿主PYTHONPATH，避免base typing_extensions污染。新的机器不能直接复制本venv当作可移植环境。

如需在本机重建：

```powershell
& 'D:\Miniconda3\envs\learn_pytorch\python.exe' -I -m venv --system-site-packages 'D:\codex学术论文2\.venv'
& 'D:\codex学术论文2\.venv\Scripts\python.exe' -I -m pip install --no-index --find-links 'D:\codex学术论文2\wheels' -r 'D:\codex学术论文2\requirements-extra.txt'
```

额外依赖的Windows/Python3.8离线轮包已保存在wheels。原模型环境为Python3.8.20、torch2.4.1+cu118、torchvision0.19.1+cu118、numpy1.24.4、Pillow10.4.0，运行前应核验。CPU无需CUDA但耗时较长。严格确定性CUDA cumsum在当前torch不支持；当前固定种子并记录告警，不保证跨设备逐位相同。

高位深/DICOM、非标准标注和EXIF方向需要显式转换方案，当前拒绝暗转。第三方数据可调用datasets.import_manifest(path,root)使用显式CSV映射；当前尚无任意上传文件自动纳入实验的界面，也不从文件名猜患者身份。

正式验证、质量界值、IACS优化/策略验证、独立外部验证、U-Net及正式亚组研究仍待后续完成，详见EXPERIMENT_PROTOCOL.md与SCI_RESULTS_REPORT.md。
