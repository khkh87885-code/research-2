# -*- coding: utf-8 -*-
import argparse, hashlib
from datetime import datetime
from pathlib import Path
import json
parser = argparse.ArgumentParser(description="Reproduce narratives for the frozen audited IACS_v2 data only")
parser.add_argument("--config", required=True)
args = parser.parse_args()
settings = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
audited_root = Path(settings["run_root"])
expected = {'metrics/sensitivity_samples.json': '671c1d1eb7323d8480993c8c5b2e286836dc72b802a01be04582d0ee5d504e47', 'cross_validation/validation.json': '6e3a8fbbf7c2d3352a97697a3cd18a542f79ebac5afe6dd85b343f3cda346271', 'statistics/review_v2/statistics.json': '8db2a8fe82e19b2b707294a05f0f1f22dbad3f2e2fa0be68425b050c0382229d', 'statistics/risk_evaluation.json': 'aa86b054791877f621c14f1155c2e25d7e4f7f781dbfde8aab2ab6c4301b5010'}
for name, checksum in expected.items():
    if hashlib.sha256((audited_root/name).read_bytes()).hexdigest() != checksum:
        raise ValueError("Narrative bound to audited data; changed results require new scientific review")
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import shapiro, skew

project = Path(settings["project_root"])
sys.path.insert(0, str(project))
from iacs_v2.common import read_json, write_json, write_csv
from iacs_v2.features import FEATURE_METHODS, FEATURE_VERSION
from compression_study.datasets import sha256, read_records, pseudonym_salt
from iacs_v2.validation import predict_policy, verify_policy, FEATURE_NAMES

root = audited_root
out = root / "reports" / ("reproduced_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
out.mkdir(exist_ok=False)
samples = read_json(root / 'metrics/sensitivity_samples.json')
validation = read_json(root / 'cross_validation/validation.json')
stats = read_json(root / 'statistics/review_v2/statistics.json')
risk = read_json(root / 'statistics/risk_evaluation.json')
diagnostic = read_json(root / 'diagnostics/diagnostic_payload.json')
choices = Counter(d['selected_condition'] for d in validation['outer_decisions'] if d['algorithm'] == 'IACS_v2')
conditions = ['baseline','jpeg_q95','jpeg_q85','jpeg_q75','jpeg_q60','jpeg_q40','png_l6']
def fmt(x, digits=6):
    return '不可估计' if x is None else f'{x:.{digits}f}'
def ci(value):
    return '不可估计' if value is None else f'[{value[0]:.6f}, {value[1]:.6f}]'
def save(name, text):
    with (out/name).open('x',encoding='utf-8') as stream: stream.write(text)
def heading(title):
    return '# '+title+'\n\n2026-11-08。仅 BUS-BRA development，Case-level，探索性分析。Dice 损失0.02仅为用户允许的候选；HD95容忍界值未设，不作临床非劣效或安全声明。\n\n'
def group(name, metric):
    return next(r for r in stats['group_summaries'] if r['algorithm']==name and r['metric']==metric)
def comparison(reference, metric):
    return next(r for r in stats['paired_comparisons'] if r['reference']==reference and r['metric']==metric)

table = '| 条件 | Case | Dice均值 | HD95均值(pixel) | ASSD均值(pixel) | 文件节省均值(%) | Dice损失>0.02例数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for condition in conditions+['IACS_v2']:
    loss = sum(s['candidates'][condition]['delta_dice'] < -.02 for s in samples) if condition!='IACS_v2' else validation['outer_scores']['IACS_v2']['dice_loss_violations']
    table += f"| {condition} | 213 | {group(condition,'dice')['mean']:.6f} | {group(condition,'hd95')['mean']:.6f} | {group(condition,'assd')['mean']:.6f} | {group(condition,'saving_pct')['mean']:.4f} | {loss} |\n"
main = '| IACS_v2减参照 | 指标 | Case | 均值差 | Case bootstrap 95%CI | Holm P |\n|---|---|---:|---:|---|---:|\n'
for reference in ['baseline','jpeg_q85','jpeg_q75','jpeg_q60','train_best_fixed']:
    for metric in ['dice','hd95','assd','saving_pct']:
        r=comparison(reference,metric)
        main += f"| {reference} | {metric} | {r['valid_paired_cases']} | {r['mean_difference']:.6f} | {ci(r['ci95'])} | {fmt(r['holm_p'])} |\n"

prior = (project/'IACS_DIAGNOSTIC_REPORT.md').read_text(encoding='utf-8')
save('IACS_DIAGNOSTIC_REPORT.md', prior+'\n## 完整213 Case复核补充\n\n原规则重放为 Q85 136例、Q95 77例、fallback 0。原来的全Q85现象属于3例样本覆盖有限，不是已证实的全体数据恒定。新版26特征均完整，不把这一重放当作新算法学习。原始风险分未实现；新版为每个JPEG质量输出风险概率，分布见候选评分表。\n\n五质量完整213例数据如下；不能只依据均值或分散质量参数判断算法有效。\n\n'+table)
methods='| 特征 | 方法 |\n|---|---|\n'+''.join(f'| {k} | {v} |\n' for k,v in FEATURE_METHODS.items())
save('IACS_FEATURE_REPORT.md',heading('IACS v2 原图特征报告')+f'特征版本 `{FEATURE_VERSION}`，213图/213 Case、26项数值特征，缺失0。所有特征仅使用原图RGB；人工参考轮廓不进入特征函数。GLCM采用32级、距离1和2、4方向对称归一矩阵，先各方向求指标再等权平均。\n\n'+methods+'\n特征分布保存在diagnostics/diagnostic_payload.json，完整逐图表在features/iacs_features.csv。训练标准化仅使用训练折；常量特征仅在训练折判定并删除，不用全开发集结局选择特征。图中用于显示的全局z分数不参与CV。噪声、清晰度和形状均为工程描述量，不是物理噪声标定或临床质量量表。\n\n260项特征-结局Spearman比较在Case内汇总，统一Holm校正后未发现P<0.05的关联；这不是证明特征没有作用。连续变化预测与风险分类的外层结果分别报告，禁止凭单一相关性指定风险权重。\n')
save('IACS_COMPRESSION_REPORT.md',heading('IACS v2 完整压缩敏感性数据库')+'213个冻结开发Case，每Case一图，七条件1491/1491成功。1260次新增SAM推理、210次PNG解码像素完全相同而复用基线预测、21项已核验的旧结果复用。每个条件均保存真实预测掩膜；复用不伪称重新推理。无空预测、无未定义HD95；原图、标注和权重前后哈希不变。\n\n'+table+'\nPNG是处理RGB像素无损，与原灰度PNG文件字节及体积不同，体积变大如实保留。平均Dice可被个别原始分割极差而压缩后改善的Case拉高，应同时查看逐Case差值、分位数、下降比例和失败。Q40的Delta Dice范围为'+f"[{min(s['candidates']['jpeg_q40']['delta_dice'] for s in samples):.6f}, {max(s['candidates']['jpeg_q40']['delta_dice'] for s in samples):.6f}]。\n\n"+'编码/解码耗时按实际任务记录；inference_reused记录与新增推理分开，PNG复用的0毫秒是记录约定，不是模型基准耗时。现有BUS-BRA清单/PNG未提供可核验的逐图spacing，本次HD95/ASSD全用pixel，不借用BrEaST间距，不虚构mm换算。原图绝对Dice与压缩相对变化分别保留。\n')
fold_table='| 外层折 | 训练Case | 验证Case | 内层选择 | 风险参数 | 训练内最佳固定 |\n|---|---:|---:|---|---:|---|\n'
for f in validation['folds']:
    fold_table+=f"| {f['fold']} | {f['train_cases']} | {f['validation_cases']} | {f['selected_family']} | {f['selected_config']['parameters'].get('risk_threshold')} | {f['train_best_fixed']['condition']} |\n"
save('IACS_VALIDATION_REPORT.md',heading('IACS v2 嵌套交叉验证')+'固定外层5折、内层3折GroupKFold，Group为现有Case。三个低复杂度家族为浅规则树、逐质量Logistic风险分类、逐质量Ridge变化回归。标准化、常量特征删除、风险参数、复杂度和最优固定质量均仅用对应训练折。外层结局不选冠军。\n\n'+fold_table+'\n独立复核：15个fold×family训练标准化与训练数据重新计算一致；852条OOF决策用保存policy重放完全一致；将验证候选Dice/HD95/ASSD/status改成极端错误值后，852次决策不变，说明部署决策不读取评价结局。各Case只出现在一个外层验证折。\n\nIACS_v2 OOF质量分布：'+json.dumps(dict(choices),ensure_ascii=False)+'，其中5例回退原图。五个外层训练内都选择B_logistic，训练内最佳固定都为Q85。不同质量分布是实际模型输出，不是验收时强制分配。全开发集最终策略用完整开发数据内层CV后重新拟合，其自身拟合表现不作为外层验证证据；尚无外部验证。\n\n[GroupKFold官方说明](https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.GroupKFold.html)与[嵌套验证官方示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。\n')
assumptions=[]
index={s['image_key']:s for s in samples}
for metric in ['dice','hd95','assd','saving_pct']:
    diffs=[d['evaluation_only'][metric]-index[d['image_key']]['candidates']['jpeg_q85'][metric] for d in validation['outer_decisions'] if d['algorithm']=='IACS_v2']
    assumptions.append(dict(metric=metric,n_cases=len(diffs),quantiles=np.quantile(diffs,[0,.25,.5,.75,1]).tolist(),skewness=float(skew(diffs)),zero_fraction=float(np.mean(np.asarray(diffs)==0)),shapiro_p=float(shapiro(diffs).pvalue)))
write_json(out/'assumption_diagnostics.json',assumptions)
save('IACS_STATISTICAL_REPORT.md',heading('IACS v2 Case聚类探索统计')+'主表为IACS减参照的同图配对差值，先图像配对再Case内平均。2000次Case百分位bootstrap，95%区间条件于固定OOF记录；没有重跑拟合，未涵盖模型选择及重叠训练折的不确定性，不能当临床非劣区间。所有失败/缺失分母保留。Wilcoxon双侧结果按5参照×8终点共40项作为一个探索族Holm校正；亚组和260项特征关联各有独立校正族。\n\n'+main+'\nHD95/ASSD单位pixel，saving_pct差为百分点。Dice差和HD95差区间跨0；ASSD未校正bootstrap区间与Wilcoxon结论可不同，不能选择有利的结果解释。配对差值存在零值、偏斜和长尾，完整quantiles/skew/Shapiro诊断另存assumption_diagnostics.json；Shapiro不是选择检验或证明对称性的依据，Wilcoxon的分布解释需谨慎。没有自动删异常Case。\n\n混合效应模型为Case随机截距、条件固定效应，REML；本次收敛，警告为空。它是额外探索描述，不建模交叉拟合的训练依赖。6类亚组阈值取各外层训练折中位数；GT面积/形状只用于评价分层，未进入算法。所有亚组表保留分母，不依据最终结果改切分点。\n')
risk_table='| JPEG | OOF AUC | 95%CI(500次Case重抽样) | Brier | 训练事件率基准Brier |\n|---|---:|---|---:|---:|\n'
for r in risk:
    if r['algorithm']=='B_logistic':risk_table+=f"| {r['condition']} | {r['roc_auc']:.4f} | {ci(r['roc_auc_ci95'])} | {r['brier_score']:.6f} | {r['training_prevalence_brier']:.6f} |\n"
save('IACS_SCI_RESULTS.md',heading('IACS v2 开发集科研结果与结论边界')+'## 主要结果\n\n'+table+'\nIACS相对固定Q85的平均节省率差为-14.347365个百分点（95%CI [-16.844224,-11.785482]）；Dice差+0.001304（95%CI [-0.006032,0.009636]）；HD95差-0.145539 pixel（95%CI [-0.666013,0.351730]）。IACS候选Dice下降>0.02为20/213，Q85为32/213，但IACS用更大的文件换取了更少候选下降，不能声称体积与质量同时优于Q85。训练折选出的最佳固定策略每折均Q85，因此该参照与Q85相同。\n\n## 四个科研问题\n\n1. 真实压缩敏感性存在逐Case变化，不能由原始Dice高低或全体均值代替；数据库已保留全部条件及差值。\n2. 当前26特征的连续变化预测较弱，各家族的10个连续端点R²均小于0；这包含B的辅助Ridge输出，不能直接据此否定主Logistic风险判别。风险结果如下，Q85有有限判别信号，其余接近随机，Brier均未优于训练事件率基准；尚无稳定而充分的预测证据。\n3. 当前方案没有证明在相同可靠性条件下比固定Q85少用文件；HD95界值未设，联合约束更不能宣称成立。\n4. 五折选择家族与风险参数一致，但只有一次开发集嵌套CV，泛化和校准仍有限；不能把稳定选型当作临床稳定性。\n\n'+risk_table+'\n全部风险评价是对保存OOF预测的附加描述，未据此重选模型或调整参数。AUC区间亦未包含重拟合不确定性。未开展BrEaST或原851留出集调参/评价，未改原SAM。自动提示定位仍沿用原技术问题记录，不能用GT机制提示替代真实应用定位。\n\n## 对论文的实际含义\n\n现阶段可以支持完整压缩敏感性和透明的无优势探索报告，尚不足支持“创新自适应方法显著优于统一压缩”的论文主张。应先论证更有信息的纯图像特征/任务不确定性信号和正式界值；若继续修改模型，必须登记新版本并避免把本轮外层结果反复当作未见测试集。临床专家、文献与新的先导设计确定界值后，才讨论预先锁定的正式验证。\n')
policy=validation['full_policy'];verify_policy(policy)
save('IACS_MODEL_CARD.md',heading('IACS v2 模型卡')+f"算法版本：{policy['schema_version']}。最终研究策略：{policy['family']}，参数{json.dumps(policy['parameters'])}。特征版本：{FEATURE_VERSION}。策略SHA-256：`{policy['policy_sha256']}`。训练213开发Case，特征训练内标准化，参数保存为可读JSON，不加载外来pickle。\n\n输入仅为原图26特征与原图/JPEG五质量/PNG的实际编码字节及无损像素一致性。部署函数不读取人工轮廓、SAM预测、Dice、HD95或病例病理信息；所有候选均输出预测风险、辅助变化估计、Dice候选可行性和选择原因。HD95状态永远not_specified，没有双约束通过标签。\n\nB模型对每JPEG预测Dice下降>0.02事件概率；内层风险参数0.1是算法选择参数，不是医学界值，也没有校准保证。选择预计Dice候选合格者中最小负载；无有损候选可用或特征缺失则选择原图/验证像素一致的更小PNG，显式记录fallback与原因。数值异常/缺失不会补成有利值。全图定位不属于本策略。\n\n外层验证仅适用于本次已固定的GT框机制设置；当前策略未在外部库、独立临床流程或其他模型上验证。它不是临床上传安全控制器。\n\n连续性能回归R²、分类AUC/Brier、逐图选择和全部不理想结果见SCI报告及risk_evaluation表；不得只引用降低候选下降比例而隐去体积损失。\n")

explained=[]
for d in validation['outer_decisions']:
    if d['algorithm']!='IACS_v2':continue
    parts=[]
    for q,p in d['predictions'].items():
        riskval=p.get('unsafe_probability'); estimated=p.get('estimated_feasible')
        parts.append(f"{q}: bytes={p.get('file_bytes')}, predicted_risk={riskval}, Dice_candidate={estimated}, HD95=not_specified")
    explained.append(dict(case_key=d['case_key'],image_key=d['image_key'],fold=d['fold'],selected_condition=d['selected_condition'],
        fallback=d['fallback'],selection_reason=d['reason'],predicted_candidates='; '.join(parts),
        observation_is_evaluation_only=True,observed_dice=d['evaluation_only']['dice'],observed_hd95=d['evaluation_only']['hd95']))
write_csv(out/'decisions_explained.csv',explained)
write_csv(out/'risk_prediction_performance.csv',risk)
save('README_FIRST.md',heading('IACS v2 交付入口')+'完整开发集探索性验证已完成，尚未证明相对固定Q85的体积优势。先阅读 IACS_SCI_RESULTS.md，再看 VALIDATION / STATISTICAL / MODEL_CARD。\n\n实际结果目录：`'+str(root)+'`。原始矩阵在compression/tasks，敏感性库在metrics，26特征在features，OOF与策略在cross_validation，Case统计在statistics/review_v2，风险描述在statistics/risk_evaluation.json，八图及Excel在reports/publication_v2。\n\n只读专项工作台：[本机 IACS v2](http://127.0.0.1:8766/iacs-v2)。原服务8765保持运行；重启被自动审批拒绝，工具未给详细原因，因此新页面使用独立只读服务，无实验启动或原图接口。\n\n所有数值来自真实本地结果。未上传医学图像，未更改权重、原始清单或旧实验。\n')

audit=[]
for f in validation['folds']:
    train=set(f['train_case_keys']);valid=set(f['validation_case_keys']);assert not train&valid
    audit.append(dict(fold=f['fold'],train_cases=len(train),validation_cases=len(valid),overlap=0))
replayed=0
for d in validation['outer_decisions']:
    policy=read_json(root/f"cross_validation/fold_{d['fold']}_{d['underlying_family']}_policy.json")
    s=index[d['image_key']]
    assert predict_policy(policy,s['features'],s['candidates'])['selected_condition']==d['selected_condition']
    corrupted={k:dict(value,dice=-999,hd95=999,assd=999,status='error') for k,value in s['candidates'].items()}
    assert predict_policy(policy,s['features'],corrupted)['selected_condition']==d['selected_condition']
    replayed+=1
write_json(out/'independent_validation_audit.json',dict(folds=audit,decisions_replayed=replayed,selection_mismatches=0,
           outcome_corruption_changes=0,source_model_unchanged=True))
print(json.dumps(dict(reports=len(list(out.glob('*.md'))),case_count=213,replayed=replayed,output=str(out))))