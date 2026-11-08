# 数据与模型可用性

## 公开材料

IACS v2 的匿名 Case/图像键、原图数值特征、文件字节数、编码参数、预测评价指标、全部候选质量、失败状态、折分、模型策略及统计结果在本仓库。冻结输入 `metrics/sensitivity_samples.json` 原字节 SHA-256 为 `671c1d1eb7323d8480993c8c5b2e286836dc72b802a01be04582d0ee5d504e47`。保留匿名相对路径及文件哈希以支持数值追溯，不包含它们指向的图像。

旧工作台原始编号记录仅发布另建的匿名副本；其生成方式及删除字段见该目录 README。其原数据库和原导出表没有上传。旧 smoke 两轮使用同 3 个 Case，不是 6 个独立 Case。当前表中的 Case 不是经核实的患者身份单位。

## 本地保留材料

原始图像、参考/预测掩膜、压缩像素、临床表格、原身份清单、私有伪名密钥和 SAM checkpoint 不再分发。重新采集指标需要用户自行按原数据条款获取资料并在本地配置。原始私有密钥无法从仓库恢复；新密钥会产生新排序和分组，不能声称只靠相同随机种子恢复原折分。精确数值复现应使用已公开的冻结匿名数据和折分。

## 原始来源

- BUS-BRA：[数据记录](https://zenodo.org/records/8231412)、[数据论文](https://doi.org/10.1002/mp.16812)。
- BrEaST：[数据记录](https://doi.org/10.7937/9WKK-Q141)、[数据论文](https://www.nature.com/articles/s41597-024-02984-z)。BrEaST 没有参与本轮 IACS v2 验证。
- SAM：[官方仓库](https://github.com/facebookresearch/segment-anything)，本次来源 commit `dca509fe793f601edb92606367a655c15ac00fdf`。ViT-B checkpoint SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`，375042383 bytes。

本仓库不替代这些来源的授权与使用条款，也不授予原始临床数据的再分发许可。
