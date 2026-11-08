# BrEaST 数据集只读审计

状态：audit_passed。正式数据问题阻断后续实验；normal计划无主病灶标注不是数据错误。

实查：{"images": 256, "main_masks": 252, "normal_without_main": 4, "other_masks": 14, "transparent_images": 31, "black_composite_changed_images": 8}。

- 原图按alpha在不透明黑背景上合成后转RGB；没有修改原图。
- 主病灶由官方工作簿精确配对；其他标注单列且不混入主参考。
- Pixel_size按官方cm单位乘10转mm，轴序为(row,column)；支持各向异性显式宽高对。
- 精确文件/可见像素查重不能证明患者来源独立，继续标记跨数据集验证。
- 所有records/CSV含本地原始Case代码与路径，仅供本机，不可原样上传。

阻断问题：0；警告：2。详细计数、来源SHA和查重结果见DATA_AUDIT.json。

来源：[原始数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10830496/)，[TCIA集合](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)；访问核对日期2026-11-08。