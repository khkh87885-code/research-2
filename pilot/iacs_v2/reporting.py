"""Exploratory Case-level reports from local, deidentified numeric observations.

This module never opens source images, masks, old experiment folders or models.
Prediction scores and outer-fold observed outcomes are explicitly distinguished.
"""
import base64
import csv
import hashlib
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NOTICE = "探索性 Case-level 分析；Dice 下降 0.02 仅为候选。HD95 容忍界值未设，不能判断双约束达标、临床安全或非劣效。"
SUCCESS = {"ok", "completed", "success"}
FIGURE_TITLES = ["研究流程（方法图，非实验结果）", "原图特征分布", "JPEG 体积与分割质量", "外层验证 IACS 与固定条件", "原图特征与决策关联", "外层交叉验证各折", "训练折阈值定义的亚组", "逐 Case 体积 / Dice / HD95 Pareto"]


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _clean(value, key=""):
    """Remove raw identities/paths and pseudonymize local display keys again."""
    if isinstance(value, dict):
        out = {}
        for name, item in value.items():
            lowered = str(name).lower()
            if lowered in {"raw_id", "raw_ids", "name", "dicom_uid", "accession_number"} or "case_id" in lowered or "image_id" in lowered or ("patient" in lowered and lowered != "patient_semantics_verified") or "path" in lowered or lowered.endswith("_directory"):
                continue
            out[str(name)] = _clean(item, lowered)
        return out
    if isinstance(value, (list, tuple, np.ndarray)):
        if key.endswith("case_keys"):
            return [_clean(item, "case_key") for item in value]
        if key.endswith("image_keys"):
            return [_clean(item, "image_key") for item in value]
        return [_clean(item, key) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if key in {"case_key", "image_key"} and value is not None:
        return key.split("_")[0] + "_" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and (":\\" in value or ":/" in value or value.startswith(("/", "\\\\"))):
        return "[本地路径不导出]"
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _observed(row, metric="dice"):
    status = str(row.get("status", "")).lower()
    return (not status or status in SUCCESS) and finite(row.get(metric)) is not None


def _mean(values):
    values = [finite(value) for value in values]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _flat(row):
    return {key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()}


def _cell(value):
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status", "reason"]
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in _flat(row).items()})
    return fields


def _tables(data):
    features = []
    for row in data.get("feature_rows", []):
        flat = {k: v for k, v in row.items() if k != "features"}
        flat.update(row.get("features", {}))
        features.append(flat)
    validation = data.get("validation") or {}
    decisions = validation.get("outer_decisions", [])
    scores = []
    outer = []
    for row in decisions:
        flat = {k: v for k, v in row.items() if k not in ("predictions", "evaluation_only", "candidate_evaluation", "subgroups", "subgroup_thresholds")}
        flat.update({"observed_" + k: v for k, v in (row.get("evaluation_only") or {}).items()})
        flat["subgroups"] = row.get("subgroups", {})
        outer.append(flat)
        for condition, score in (row.get("predictions") or {}).items():
            scores.append(dict(case_key=row.get("case_key"), image_key=row.get("image_key"),
                               fold=row.get("fold"), algorithm=row.get("algorithm"), condition=condition,
                               hd95_constraint="not_specified", interpretation="predicted_exploratory_Dice_candidate_only", **score))
    supplied = data.get("statistics") or {}
    associations = supplied.get("feature_associations") or []
    feature_index = {row.get("image_key"): row.get("features", {}) for row in data.get("feature_rows", [])}
    from scipy.stats import spearmanr
    by_condition = defaultdict(list)
    for row in data.get("compression_rows", []):
        if _observed(row) and finite(row.get("delta_dice")) is not None:
            by_condition[row.get("condition", "unknown")].append(row)
    names = sorted({key for row in feature_index.values() for key in row if finite(row[key]) is not None})
    for condition, rows in ([] if associations else by_condition.items()):
        for name in names:
            pairs = [(finite(feature_index.get(row.get("image_key"), {}).get(name)), finite(row.get("delta_dice"))) for row in rows]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            rho = None
            if len(pairs) >= 3 and len({p[0] for p in pairs}) > 1 and len({p[1] for p in pairs}) > 1:
                rho = finite(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
            associations.append(dict(condition=condition, feature=name, n_cases=len(pairs), spearman_rho=rho,
                                     interpretation="full_development_descriptive_not_model_selection"))
    def records(value):
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [dict(item, group=key) if isinstance(item, dict) else {"group": key, "value": item} for key, item in value.items()]
        return []
    return {"features": features, "compression": data.get("compression_rows", []), "candidate_scores": scores,
            "outer_decisions": outer, "folds": validation.get("folds", []), "feature_associations": records(associations),
            "group_summaries": records(supplied.get("group_summaries")), "paired_comparisons": records(supplied.get("paired_comparisons")),
            "subgroups": records(supplied.get("subgroups")), "mixed_effects": records(supplied.get("mixed_effects")),
            "prediction_performance": records(supplied.get("prediction_performance"))}


def _pending(ax, message):
    ax.axis("off")
    ax.text(.5, .5, "待完成 / pending\n" + message, transform=ax.transAxes, ha="center", va="center", wrap=True, fontsize=12, color="#795527")


def _make_figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = "Microsoft YaHei" if "Microsoft YaHei" in fonts else "DejaVu Sans"
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555", "#56B4E9"]
    rows = data.get("compression_rows", [])
    feature_rows = data.get("feature_rows", [])
    decisions = (data.get("validation") or {}).get("outer_decisions", [])
    v2 = [r for r in decisions if r.get("algorithm") == "IACS_v2"]
    assets = []
    with plt.rc_context({"font.family": font, "axes.unicode_minus": False, "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}):
        for index, title in enumerate(FIGURE_TITLES, 1):
            fig = plt.figure(figsize=(11, 6.3), constrained_layout=True)
            status = "observed"
            note = "描述性 Case-level 观测；缺失不填补。"
            if index == 1:
                ax = fig.add_subplot(111)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
                blocks = ["冻结 BUS-BRA 开发 Case\n原图特征 + 七条件实测", "外层 Case 分组交叉验证\n内层选择仅使用训练折", "逐候选预测 Dice / HD95 变化\nHD95 容忍界值未设", "选择探索性 Dice 候选\n无候选保守回退并记录原因", "外层留出 Case 实测评价\n失败分母 + 不确定性 + 限制"]
                for j, text in enumerate(blocks):
                    y = .83 - j * .17
                    ax.add_patch(FancyBboxPatch((.12, y - .055), .76, .11, boxstyle="round,pad=.01", facecolor="#EAF3F6", edgecolor="#376A8C"))
                    ax.text(.5, y, text, ha="center", va="center")
                    if j < 4:
                        ax.annotate("", xy=(.5, y - .112), xytext=(.5, y - .069), arrowprops={"arrowstyle": "->", "color": "#376A8C"})
                status = "method_only"; note = "方法流程图；不表示各步骤已获得有效研究结论。"
            elif index == 2:
                keys = sorted({k for r in feature_rows for k, value in r.get("features", {}).items() if finite(value) is not None})
                preferred = [k for k in keys if any(term in k for term in ("noise", "clar", "sharp", "contrast", "entropy", "edge"))]
                keys = (preferred + [k for k in keys if k not in preferred])[:6]
                axes = fig.subplots(2, 3).ravel()
                if not keys:
                    status = "pending"
                for j, ax in enumerate(axes):
                    if j >= len(keys):
                        _pending(ax, "尚无对应真实特征"); continue
                    values = [finite(r.get("features", {}).get(keys[j])) for r in feature_rows]
                    values = [v for v in values if v is not None]
                    ax.hist(values, bins=min(15, max(1, int(np.sqrt(len(values))))), color=palette[j], alpha=.85)
                    ax.set(title=keys[j] + " (n=" + str(len(values)) + ")", ylabel="Case count")
                note = "原图特征的实际分布；固定等宽直方图，未按结局筛图。"
            elif index == 3:
                axes = fig.subplots(1, 3)
                groups = defaultdict(list)
                for row in rows:
                    if "jpeg" in str(row.get("condition", "")).lower() and _observed(row):
                        groups[row["condition"]].append(row)
                if not groups:
                    status = "pending"
                    for ax in axes: _pending(ax, "尚无 JPEG 真实指标")
                for j, (name, members) in enumerate(sorted(groups.items())):
                    for ax, metric in zip(axes, ["dice", "hd95", "assd"]):
                        pairs = [(finite(r.get("output_bytes")), finite(r.get(metric))) for r in members]
                        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                        ax.scatter([x / 1024 for x, y in pairs], [y for x, y in pairs], s=16,
                                   alpha=.65, marker=["o", "s", "^", "D", "v"][j % 5], color=palette[j % 7], label=name + " n=" + str(len(pairs)))
                for ax, metric in zip(axes, ["Dice", "HD95 (pixel)", "ASSD (pixel)"]):
                    ax.set(xlabel="Final payload (KiB)", ylabel=metric); ax.grid(alpha=.15)
                if groups: axes[0].legend(fontsize=6)
            elif index == 4:
                axes = fig.subplots(1, 3)
                series = defaultdict(list)
                valid_keys = {r.get("image_key") for r in v2}
                for row in rows:
                    condition = str(row.get("condition", ""))
                    if row.get("image_key") in valid_keys and (condition == "baseline" or "jpeg" in condition.lower() or "png" in condition.lower()):
                        series[condition].append(row)
                for row in v2: series["IACS_v2 (outer OOF)"].append(row.get("evaluation_only") or {})
                if not v2:
                    status = "pending"
                    for ax in axes: _pending(ax, "尚无外层验证决策；不以训练表现代替")
                else:
                    names = sorted(series)
                    for ax, metric in zip(axes, ["dice", "saving_pct", "hd95"]):
                        for j, name in enumerate(names):
                            values = [finite(r.get(metric)) for r in series[name] if _observed(r, metric)]
                            values = [v for v in values if v is not None]
                            if values:
                                ax.scatter([j] * len(values), values, s=8, alpha=.16, color=palette[j % 7])
                                ax.scatter(j, np.mean(values), s=45, marker="D", color=palette[j % 7])
                                ax.annotate("n=%d" % len(values), (j, np.mean(values)), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6)
                        ax.set_xticks(range(len(names)), names, rotation=65, ha="right", fontsize=6)
                        ax.set_ylabel({"dice": "Observed Dice", "saving_pct": "Saving (%)", "hd95": "HD95 (pixel)"}[metric]); ax.grid(axis="y", alpha=.15)
                note = "每点为相同外层留出 Case 的实测结局，菱形为均值；不据此事后选全局冠军。失败保留在表中。"
            elif index == 5:
                ax = fig.add_subplot(111)
                feature_index = {r.get("image_key"): r.get("features", {}) for r in feature_rows}
                names = sorted({k for values in feature_index.values() for k, v in values.items() if finite(v) is not None})[:12]
                conditions = sorted({str(r.get("selected_condition")) for r in v2})
                if not names or not conditions:
                    status = "pending"; _pending(ax, "尚无真实 v2 决策与匹配特征")
                else:
                    stats = {}
                    for name in names:
                        values = [finite(f.get(name)) for f in feature_index.values()]; values = [v for v in values if v is not None]
                        stats[name] = (np.mean(values), np.std(values)) if values else (0, 0)
                    matrix = np.full((len(conditions), len(names)), np.nan)
                    for i, condition in enumerate(conditions):
                        for j, name in enumerate(names):
                            values = [finite(feature_index.get(r.get("image_key"), {}).get(name)) for r in v2 if str(r.get("selected_condition")) == condition]
                            values = [v for v in values if v is not None]
                            mean, sd = stats[name]
                            if values: matrix[i, j] = (np.mean(values) - mean) / sd if sd > 0 else 0
                    plot = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
                    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
                    ax.set_yticks(range(len(conditions)), conditions)
                    fig.colorbar(plot, ax=ax, label="Descriptive feature z score (display clipped ±2)")
                note = "按外层决策分组的原图特征均值。全开发集标准化仅用于显示，不用于拟合或交叉验证；关联不代表因果。"
            elif index == 6:
                axes = fig.subplots(1, 2)
                groups = defaultdict(list)
                for row in decisions:
                    groups[(str(row.get("algorithm")), row.get("fold"))].append(row.get("evaluation_only") or {})
                algorithms = sorted({k[0] for k in groups})
                if not groups:
                    status = "pending"
                    for ax in axes: _pending(ax, "外层各折真实验证表现待完成")
                for j, algorithm in enumerate(algorithms):
                    folds = sorted({k[1] for k in groups if k[0] == algorithm}, key=str)
                    for ax, metric in zip(axes, ["delta_dice", "saving_pct"]):
                        values = [_mean(r.get(metric) for r in groups[(algorithm, fold)] if _observed(r, metric)) for fold in folds]
                        ax.plot(range(len(folds)), [np.nan if v is None else v for v in values], marker=["o", "s", "^", "D"][j % 4], color=palette[j % 7], label=algorithm)
                        ax.set_xticks(range(len(folds)), [str(f) for f in folds]); ax.set_xlabel("Outer validation fold")
                        ax.set_ylabel("Observed Dice change" if metric == "delta_dice" else "Observed saving (%)"); ax.grid(alpha=.15)
                if groups: axes[0].legend(fontsize=7)
                note = "仅外层留出 Case 的均值；训练拟合描述与内层参数选择在折表另列，不能作为验证表现。"
            elif index == 7:
                axes = fig.subplots(1, 2)
                groups = defaultdict(list)
                for row in v2:
                    for key, value in (row.get("subgroups") or {}).items():
                        groups[str(key) + "=" + str(value)].append(row.get("evaluation_only") or {})
                if not groups:
                    status = "pending"
                    for ax in axes: _pending(ax, "尚无按训练折阈值定义的外层亚组")
                else:
                    names = sorted(groups)
                    for ax, metric in zip(axes, ["delta_dice", "saving_pct"]):
                        for j, name in enumerate(names):
                            values = [finite(r.get(metric)) for r in groups[name] if _observed(r, metric)]
                            values = [v for v in values if v is not None]
                            if values: ax.scatter(np.mean(values), j, marker="D", color=palette[j % 7])
                        counts = [sum(_observed(r, metric) for r in groups[name]) for name in names]
                        ax.set_yticks(range(len(names)), [name + " n=%d/%d" % (n, len(groups[name])) for name, n in zip(names, counts)], fontsize=7)
                        ax.set_xlabel("Observed Dice change" if metric == "delta_dice" else "Observed saving (%)"); ax.grid(axis="x", alpha=.15)
                note = "探索性亚组；阈值必须来自相应训练折。各亚组可能交叠，不能累加为独立 Case。"
            else:
                ax = fig.add_subplot(111)
                valid = [r for r in rows if _observed(r) and finite(r.get("output_bytes")) is not None and finite(r.get("hd95")) is not None]
                if not valid:
                    status = "pending"; _pending(ax, "尚无三目标完整实测记录")
                else:
                    values = np.asarray([[float(r["output_bytes"]) / 1024, float(r["dice"]), float(r["hd95"])] for r in valid])
                    plot = ax.scatter(values[:, 0], values[:, 1], c=values[:, 2], cmap="viridis", s=22, alpha=.6)
                    frontier = []
                    groups = defaultdict(list)
                    for j, row in enumerate(valid): groups[row.get("image_key")].append(j)
                    for indexes in groups.values():
                        costs = values[indexes] * [1, -1, 1]
                        for local, actual in enumerate(indexes):
                            dominated = np.any(np.all(costs <= costs[local], axis=1) & np.any(costs < costs[local], axis=1))
                            if not dominated: frontier.append(actual)
                    if frontier: ax.scatter(values[frontier, 0], values[frontier, 1], facecolors="none", edgecolors="#222222", s=50, linewidths=.6, label="Within-Case nondominated")
                    ax.set(xlabel="Final payload (KiB)", ylabel="Observed Dice"); ax.legend(fontsize=8)
                    fig.colorbar(plot, ax=ax, label="HD95 (pixel)")
                note = "逐 Case 三目标非支配点以空心圈标记；不跨 Case 选择最优图像，不将 Pareto 位置称为安全。缺 HD95 的记录仍在表中。"
            fig.suptitle("Figure %d · %s" % (index, title), fontsize=13)
            fig.supxlabel(textwrap.fill(note, 70) + "\n探索性 / Case-level；HD95 容忍界值未设。", fontsize=8)
            stem = "figure_%02d" % index
            paths = {}
            for extension in ("png", "svg", "pdf"):
                path = output / (stem + "." + extension)
                fig.savefig(path, dpi=300, facecolor="white")
                paths[extension] = path.name
            plt.close(fig)
            assets.append(dict(number=index, title=title, status=status, note=note, files=paths))
    return assets


def _html(report, assets):
    # The full report.json retains provenance and raw validation structures.
    # The browser only needs the display tables and their interpretation.
    browser_report = {key: report[key] for key in (
        "schema_version", "notice", "analysis_unit", "summary", "diagnostics", "tables",
        "limitations", "hd95_margin", "dice_candidate_margin") if key in report}
    browser_report["snapshot_type"] = "interface_data_snapshot_not_full_report"
    encoded = json.dumps(browser_report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    embedded = json.dumps(assets, ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'">
<title>IACS_v2 · 开发集探索工作台</title><style>
:root{--ink:#173a42;--muted:#5c7375;--accent:#167867;--line:#dbe5e2}*{box-sizing:border-box}body{margin:0;background:#f3f6f5;color:var(--ink);font:14px/1.6 "Microsoft YaHei","Segoe UI",sans-serif}header{padding:24px 4vw;background:#123b42;color:white}header a{color:#c9ede3}h1{font-size:25px;margin:4px 0}main{max-width:1500px;margin:auto;padding:24px}nav{display:flex;gap:7px;flex-wrap:wrap;margin:16px 0}button,select,input{font:inherit;padding:9px 13px;border:1px solid var(--line);border-radius:6px;background:white;color:var(--ink)}button{cursor:pointer}button.active{background:var(--accent);color:white}section.panel{display:none}section.panel.active{display:block}.card,.box{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:18px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.card{padding:14px}.card strong{display:block;font-size:25px}.muted{color:var(--muted);font-size:12px}.notice{background:#fff3dc;border-left:4px solid #ae7a27;padding:15px;margin:16px 0}.table{overflow:auto;max-height:560px;border:1px solid var(--line)}table{border-collapse:collapse;font-size:12px;width:100%;white-space:nowrap}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}th{background:#edf4f1;position:sticky;top:0}.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0}pre{white-space:pre-wrap;word-break:break-word;max-height:390px;overflow:auto;background:#eff4f1;padding:14px;font-size:12px}.figure{width:100%;height:auto;border:1px solid var(--line)}h2{font-size:19px}h3{font-size:15px}.downloads a{display:inline-block;margin:6px;color:var(--accent)}.figuregrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}label{font-size:12px;color:var(--muted)}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}.figuregrid{grid-template-columns:1fr}main{padding:14px}}a{color:var(--accent)}</style></head>
<body><header><a href="/">← 原八页科研工作台</a><div class="muted" style="color:#c0d9d2">IACS_v2 · 本机自包含结果快照</div><h1>从原算法诊断到逐候选探索</h1><p>仅 BUS-BRA development · Case-level · 无医学图像和外部资源</p></header><main>
<div class="notice">探索性分析：Dice 下降 0.02 仅为候选；HD95 容忍界值未设。不能判断双约束达标、临床安全或非劣效。</div>
<div id="counts" class="cards"></div><nav id="tabs"><button data-page="diagnosis" class="active">算法诊断</button><button data-page="quality">质量比较</button><button data-page="adaptive">自适应候选</button><button data-page="validation">CV 模型与关联</button><button data-page="figures">SCI 图表</button></nav>
<section class="panel active" id="diagnosis"><div class="box"><h2>原算法与 v2 诊断</h2><p class="muted">原规则普通 Q85 分支与 fallback 分开报告；不同图像无需人为分散质量。下面只显示本快照真实诊断。</p><pre id="diagnostics"></pre></div><div class="box"><h2>原图特征</h2><div class="row"><input id="feature-search" placeholder="按匿名 Case / 图像键筛选"><button data-download="features">下载特征 CSV</button></div><div id="feature-table"></div></div></section>
<section class="panel" id="quality"><div class="box"><h2>五档 JPEG 与对照实测</h2><p class="muted">保留失败分母；实际 Dice / HD95 / ASSD 与字节数不由模型预测代替。距离单位 pixel。</p><div class="row"><label>处理条件 <select id="quality-condition"></select></label><label>状态 <select id="quality-status"></select></label><button data-download="compression">下载全部质量记录 CSV</button></div><div id="quality-count" class="muted"></div><div id="quality-table"></div></div><div class="box"><h2>Case 汇总与配对探索统计</h2><div id="groups-table"></div><div id="paired-table"></div><button data-download="group_summaries">下载条件汇总 CSV</button> <button data-download="paired_comparisons">下载配对比较 CSV</button><p class="muted">区间、Case bootstrap 与 Holm 校正以统计表实际字段为准；缺失不作通过判断。</p></div></section>
<section class="panel" id="adaptive"><div class="box"><h2>逐图、逐算法的外层留出决策</h2><p class="muted">候选分数来自训练折模型，evaluation_only 才是外层验证实测。满足预测 Dice 候选不代表 HD95 约束通过。回退及原因完整保留。</p><div class="row"><label>算法 <select id="algorithm"></select></label><input id="case-search" placeholder="匿名 Case / 图像键"><button data-download="outer_decisions">下载决策 CSV</button></div><div id="decision-count" class="muted"></div><div id="decision-table"></div></div><div class="box"><h2>逐候选评分（预测）</h2><div id="scores-table"></div><button data-download="candidate_scores">下载候选评分 CSV</button></div></section>
<section class="panel" id="validation"><div class="box"><h2>外层各折、内层选择与训练描述</h2><p class="muted">train_fit_descriptive 为训练拟合描述，不能当作验证表现；outer_decisions 来自外层留出 Case。各折参数、分母和警告均保留。</p><div id="fold-table"></div><button data-download="folds">下载各折记录 CSV</button></div><div class="box"><h2>模型预测误差（外层 Case OOF）</h2><p class="muted">与相应训练折均值预测比较；负 R²、缺失和较差表现均保留，不依据外层结果选择冠军。</p><div id="prediction-table"></div><button data-download="prediction_performance">下载模型误差 CSV</button></div><div class="box"><h2>原图特征与 Dice 变化的描述关联</h2><p class="muted">优先使用 Case 汇总的 Spearman / Holm 统计表；无该表时仅显示 description-only 回退关联，不作推断或 CV 参数选择。常量或不足数据返回缺失。</p><div id="association-table"></div><button data-download="feature_associations">下载关联 CSV</button></div><div class="box"><h2>训练折阈值亚组与额外模型</h2><div id="subgroup-table"></div><div id="mixed-table"></div><button data-download="subgroups">下载亚组 CSV</button> <button data-download="mixed_effects">下载额外模型 CSV</button></div></section>
<section class="panel" id="figures"><div class="box"><h2>八张可追溯图表</h2><p class="muted">方法图与结果图区分；无数据的图明确 pending。PNG 300 dpi，SVG/PDF 供后续版面编辑，未声称符合某一期刊的具体规范。</p><button id="download-json">下载界面数据 JSON</button><p class="muted">该 JSON 为界面数据快照；完整追溯数据保留在独立 report.json，可经报告清单获取。</p><div id="figure-grid" class="figuregrid"></div></div></section>
<script type="application/json" id="report-data">__REPORT__</script><script type="application/json" id="figure-data">__FIGURES__</script><script>
'use strict';const data=JSON.parse(document.getElementById('report-data').textContent),figures=JSON.parse(document.getElementById('figure-data').textContent),tables=data.tables||{};const $=id=>document.getElementById(id);let visibleDecisions=[];
const show=x=>x===null||x===undefined?'—':typeof x==='object'?JSON.stringify(x):typeof x==='number'?String(Number(x.toFixed(6))):String(x);
const labels={case_key:'匿名 Case',image_key:'匿名图像',condition:'处理条件',status:'状态',algorithm:'算法',fold:'外层折',selected_condition:'选择条件',reason:'解释',fallback:'是否回退',observed_status:'实测状态',observed_dice:'实测 Dice',observed_hd95:'实测 HD95(pixel)',observed_assd:'实测 ASSD(pixel)',observed_saving_pct:'实测节省(%)',observed_delta_dice:'实测 Dice 变化',output_bytes:'最终字节',saving_pct:'节省(%)',delta_dice:'Dice 变化',delta_hd95:'HD95 变化',predicted_delta_dice:'预测 Dice 变化',predicted_delta_hd95:'预测 HD95 变化',p_risk:'风险概率',unsafe_probability:'风险概率',estimated_feasible:'预测 Dice 候选',HD95_status:'HD95 界值状态',hd95_constraint:'HD95 约束状态',train_cases:'训练 Case',validation_cases:'验证 Case',selected_family:'训练内选择模型',selected_config:'训练内选择参数',train_best_fixed:'训练内最佳固定条件',train_fit_descriptive:'训练拟合描述(非验证)',outer_scores:'外层验证表现',feature:'特征',outcome:'结局',n_cases:'有效 Case',spearman_rho:'Spearman rho',p_value:'原始 P',holm_p:'Holm P',endpoint:'终点',valid_cases:'有效 Case',training_mean_rmse:'训练均值基准 RMSE',interpretation:'解释'};
function table(id,rows,columns){const root=$(id);root.replaceChildren();if(!rows.length){root.textContent='待完成：没有匹配的真实记录。';return;}const pageSize=200;let page=0;const fields=columns||[...new Set(rows.flatMap(r=>Object.keys(r)))],controls=document.createElement('div'),info=document.createElement('span'),prev=document.createElement('button'),next=document.createElement('button'),wrap=document.createElement('div');controls.className='row';prev.textContent='上一页';next.textContent='下一页';controls.append(prev,info,next);wrap.className='table';root.append(controls,wrap);function render(){wrap.replaceChildren();const start=page*pageSize,end=Math.min(rows.length,start+pageSize),t=document.createElement('table'),head=document.createElement('tr');fields.forEach(k=>{const el=document.createElement('th');el.textContent=labels[k]||k;head.append(el)});t.append(head);for(const row of rows.slice(start,end)){const tr=document.createElement('tr');fields.forEach(k=>{const td=document.createElement('td'),text=show(row[k]);if(text.length>350){const detail=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=text.slice(0,110)+'…（展开）';pre.textContent=text;detail.append(summary,pre);td.append(detail)}else td.textContent=text;tr.append(td)});t.append(tr)}wrap.append(t);info.textContent='显示 '+(start+1)+'–'+end+' / '+rows.length+' 条；每页200，CSV下载保留完整记录。';prev.disabled=page===0;next.disabled=end>=rows.length;}prev.onclick=()=>{page--;render()};next.onclick=()=>{page++;render()};render();}
function options(id,rows,key){const el=$(id);el.append(new Option('全部',''));for(const value of [...new Set(rows.map(r=>String(r[key]??'')))].sort())if(value)el.append(new Option(value,value));}
function filterQuality(){const rows=(tables.compression||[]).filter(r=>(!$('quality-condition').value||String(r.condition)===$('quality-condition').value)&&(!$('quality-status').value||String(r.status)===$('quality-status').value));$('quality-count').textContent='当前 '+rows.length+' 条；失败与缺失均保留。';table('quality-table',rows,['case_key','image_key','condition','status','dice','hd95','assd','output_bytes','saving_pct','delta_dice','delta_hd95']);}
function filterDecisions(){const search=$('case-search').value.toLowerCase(),algorithm=$('algorithm').value;visibleDecisions=(tables.outer_decisions||[]).filter(r=>(!algorithm||r.algorithm===algorithm)&&(!search||String(r.case_key+' '+r.image_key).toLowerCase().includes(search)));$('decision-count').textContent='当前 '+visibleDecisions.length+' 条决策；fallback '+visibleDecisions.filter(r=>r.fallback).length+' 条。HD95 容忍界值未设。';table('decision-table',visibleDecisions,['case_key','image_key','fold','algorithm','selected_condition','reason','fallback','observed_status','observed_dice','observed_hd95','observed_assd','observed_saving_pct','observed_delta_dice']);const identities=new Set(visibleDecisions.map(r=>JSON.stringify([r.case_key,r.image_key,r.fold,r.algorithm])));table('scores-table',(tables.candidate_scores||[]).filter(r=>identities.has(JSON.stringify([r.case_key,r.image_key,r.fold,r.algorithm]))),['case_key','image_key','fold','algorithm','condition','predicted_delta_dice','predicted_delta_hd95','p_risk','unsafe_probability','estimated_feasible','HD95_status','hd95_constraint']);}
document.querySelectorAll('#tabs button').forEach(button=>button.onclick=()=>{document.querySelectorAll('#tabs button').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('section.panel').forEach(s=>s.classList.toggle('active',s.id===button.dataset.page));});
function save(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csv(rows){const fields=[...new Set(rows.flatMap(r=>Object.keys(r)))];const quote=x=>{let s=typeof x==='object'&&x!==null?JSON.stringify(x):String(x??'');if(/^[=+@\\-\\t\\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};return '\\ufeff'+[fields,...rows.map(r=>fields.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\\r\\n')}
document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>save(csv(tables[b.dataset.download]||[]),'IACS_v2_'+b.dataset.download+'.csv','text/csv;charset=utf-8'));$('download-json').onclick=()=>save(JSON.stringify(data,null,2),'IACS_v2_interface_snapshot.json','application/json');
const summary=data.summary||{};for(const [label,value]of [['开发 Case',summary.cases||0],['压缩记录',summary.compression_records||0],['有效 Dice',summary.valid_dice_records||0],['失败/无效',summary.failed_or_missing_records||0],['外层验证 Case',summary.outer_cases||0]]){const card=document.createElement('div');card.className='card';card.textContent=label;const strong=document.createElement('strong');strong.textContent=value;card.append(strong);$('counts').append(card)}
$('diagnostics').textContent=JSON.stringify({original_diagnosis:data.diagnostics||{status:'pending'},v2_outer_decisions:{counts:summary.v2_condition_counts||{},fallback:summary.v2_fallback_decisions??null,outer_cases:summary.outer_cases||0,hd95_constraint:'not_specified'}},null,2);options('quality-condition',tables.compression||[],'condition');options('quality-status',tables.compression||[],'status');options('algorithm',tables.outer_decisions||[],'algorithm');$('quality-condition').onchange=filterQuality;$('quality-status').onchange=filterQuality;$('algorithm').onchange=filterDecisions;$('case-search').oninput=filterDecisions;$('feature-search').oninput=()=>{const q=$('feature-search').value.toLowerCase();table('feature-table',(tables.features||[]).filter(r=>String(r.case_key+' '+r.image_key).toLowerCase().includes(q)))};table('feature-table',tables.features||[]);filterQuality();filterDecisions();table('fold-table',tables.folds||[],['fold','train_cases','validation_cases','selected_family','selected_config','train_best_fixed','train_fit_descriptive','outer_scores']);table('association-table',tables.feature_associations||[],['condition','feature','outcome','n_cases','spearman_rho','p_value','holm_p','interpretation']);table('groups-table',tables.group_summaries||[]);table('paired-table',tables.paired_comparisons||[]);table('subgroup-table',tables.subgroups||[]);table('mixed-table',tables.mixed_effects||[]);table('prediction-table',tables.prediction_performance||[],['algorithm','condition','endpoint','valid_cases','mae','rmse','training_mean_rmse','r2','spearman_rho']);
for(const figure of figures){const box=document.createElement('div');box.className='box';const h=document.createElement('h3');h.textContent='Figure '+figure.number+' · '+figure.title+' · '+figure.status;box.append(h);if(figure.png){const img=document.createElement('img');img.className='figure';img.alt=figure.title+'；'+figure.note;img.src=figure.png;box.append(img)}const note=document.createElement('p');note.className='muted';note.textContent=figure.note;box.append(note);const links=document.createElement('div');links.className='downloads';for(const format of ['png','svg','pdf'])if(figure[format]){const a=document.createElement('a');a.href=figure[format];a.download='figure_'+String(figure.number).padStart(2,'0')+'.'+format;a.textContent=format.toUpperCase();links.append(a)}box.append(links);$('figure-grid').append(box)}
</script></main></body></html>'''.replace("__REPORT__", encoded).replace("__FIGURES__", embedded)


def build_report(payload, output_dir):
    """Create one immutable report directory; refuse any pre-existing destination."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Report output exists; choose a new report version")
    for source in list(payload.get("feature_rows", [])) + list(payload.get("compression_rows", [])):
        if source.get("split") not in (None, "development") or source.get("dataset") not in (None, "BUS-BRA", "BUSBRA"):
            raise ValueError("IACS_v2 reports accept BUS-BRA development observations only")
    data = _clean(payload)
    data["notice"] = NOTICE
    data["analysis_unit"] = "Case-level"
    data["hd95_margin"] = None
    data["dice_candidate_margin"] = .02
    data["tables"] = _tables(data)
    rows = data.get("compression_rows", [])
    cases = {r.get("case_key") for r in data.get("feature_rows", []) + rows if r.get("case_key")}
    valid = sum(_observed(row) for row in rows)
    outer = (data.get("validation") or {}).get("outer_decisions", [])
    data["summary"] = dict(cases=len(cases), compression_records=len(rows), valid_dice_records=valid,
                           failed_or_missing_records=len(rows) - valid,
                           status_counts=dict(Counter(str(row.get("status", "not_recorded")) for row in rows)),
                           outer_decisions=len(outer), outer_cases=len({r.get("case_key") for r in outer if r.get("case_key")}),
                           fallback_decisions=sum(bool(r.get("fallback")) for r in outer),
                           outer_folds=len((data.get("validation") or {}).get("folds", [])))
    v2 = [r for r in outer if r.get("algorithm") == "IACS_v2"]
    data["summary"]["v2_condition_counts"] = dict(Counter(str(r.get("selected_condition")) for r in v2))
    data["summary"]["v2_fallback_decisions"] = sum(bool(r.get("fallback")) for r in v2)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    from openpyxl import Workbook
    workbook = Workbook(); workbook.remove(workbook.active)
    for name, records in data["tables"].items():
        path = output / (name + ".csv")
        fields = _write_csv(path, records); files.append(path.name)
        sheet = workbook.create_sheet(name[:31]); sheet.append(fields)
        for row in records:
            row = _flat(row); sheet.append([_cell(row.get(field)) for field in fields])
        sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
    xlsx = output / "IACS_v2_results.xlsx"; workbook.save(xlsx); files.append(xlsx.name)
    assets = _make_figures(data, output)
    embedded = []
    mimes = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    for asset in assets:
        item = {k: v for k, v in asset.items() if k != "files"}
        for extension, name in asset["files"].items():
            files.append(name)
            item[extension] = "data:" + mimes[extension] + ";base64," + base64.b64encode((output / name).read_bytes()).decode("ascii")
        embedded.append(item)
    data["figures"] = assets
    report = output / "report.json"; report.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"); files.append(report.name)
    dashboard = output / "dashboard.html"; dashboard.write_text(_html(data, embedded), encoding="utf-8"); files.append(dashboard.name)
    status = output / "REPORT_STATUS.md"
    status.write_text("# IACS_v2 探索性报告\n\n" + NOTICE + "\n\n" + json.dumps(data["summary"], ensure_ascii=False, indent=2)
                      + "\n\n失败和缺失保留在原始分母。训练描述、候选预测与外层观测不混用。所有图为本地聚合图，不含医学图像。\n\n"
                      + "\n".join("- Figure %d %s: %s" % (a["number"], a["title"], a["status"]) for a in assets), encoding="utf-8")
    files.append(status.name)
    manifest = {"schema_version": "IACS_v2_report_1", "analysis_unit": "Case-level", "notice": NOTICE,
                "summary": data["summary"], "figures": assets,
                "files": [{"name": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), "bytes": (output / name).stat().st_size} for name in files]}
    manifest_path = output / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "dashboard": str(dashboard), "report": str(report), "manifest": str(manifest_path), "xlsx": str(xlsx), "figures": assets}
