#!/usr/bin/env python3
"""Generate the Chinese semantic-v2 report from delivered JSON, never from guessed metrics.

Default: require every declared checkpoint/representation and production result.
--check-only validates inputs without publishing. --allow-partial visibly records
missing files; it does not excuse malformed metrics or inconsistent provenance.
No GPU, network, photograph, checkpoint tensor or ground-truth mask is read.

可复现运行（Python 3.10+；默认目录根据本脚本位置解析，与当前工作目录无关）：
    python /path/to/splat-studio/benchmarks/generate_semantic_v2_report.py --check-only
    python /path/to/splat-studio/benchmarks/generate_semantic_v2_report.py
输出为 benchmarks/SEMANTIC_V2_REPORT.zh-CN.md。
matplotlib 是可选依赖：安装后生成静态曲线；未安装则保留完整表格并注明未画图。
也可显式使用 --skip-plot；仅验证、读取JSON与生成Markdown不需要GPU或matplotlib。

结果解压布局（以项目根 splat-studio/ 为基准）：
    benchmarks/results/semantic-v2/semantic-v2-results/...
    benchmarks/results/semantic-v2/semantic-v2b-results/...
    benchmarks/results/semantic-v2/semantic-v2-finalization/...
    benchmarks/results/semantic-v2/semantic-worker-cuda-validation/...
    benchmarks/results/semantic-v2/semantic-worker-fresh-validation/...
    benchmarks/results/semantic-v2/DELIVERY_MANIFEST.json
    benchmarks/results/a100/benchmark-results/summary.json
    benchmarks/results/a100/full-resolution-results/summary.json
    benchmarks/results/a100/semantic-results/...
    benchmarks/results/a100/teatime-priority/...
预览归档的根清单应保留为 PREVIEW_DELIVERY_MANIFEST.json，避免覆盖完整归档清单；
两个归档如含不同时间的状态快照，遵循 docs/ 内逐文件验收记录的路径映射，勿覆盖。
需同时保留源包中的 docs/ 实验协议、测试和界面验收JSON。源码包不内嵌训练NPZ；
本脚本从交付JSON核验全部评价，另按检查点清单报告完整归档选中NPZ的本地存在数。
如用 --input 指定其他语义结果根目录，历史A100与docs依据仍从本脚本所在项目读取。
历史结果、评价尺寸、失败与缺项不能由缺失文件推断；--allow-partial会明确标示缺项。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
STEPS = (0, 400, 1200, 2400)
REPRESENTATIONS = ("soft", "editable_binary", "hierarchy_closed_binary")
REP_NAMES = {"soft": "soft", "editable_binary": "binary", "hierarchy_closed_binary": "closed"}
CASES = (
    ("semantic-v2-results", "simple-8", "balanced", "R1"),
    ("semantic-v2-results", "improved-8", "balanced", "R1"),
    ("semantic-v2-results", "improved-24", "balanced", "R1"),
    ("semantic-v2-results", "improved-24-fine", "fine", "R1"),
    ("semantic-v2b-results", "aligned-improved-24", "balanced", "R2"),
    ("semantic-v2b-results", "aligned-improved-24-fine", "fine", "R2"),
)


def number(value, digits=2):
    return "—" if value is None else f"{float(value):,.{digits}f}"


def percent(value, digits=3):
    return "—" if value is None else number(100 * float(value), digits)


def gib(value):
    return "—" if value is None else number(float(value) / 1024**3, 3)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


class Report:
    def __init__(self, inputs, output, project=PROJECT):
        self.inputs, self.output, self.project = Path(inputs).resolve(), Path(output).resolve(), Path(project).resolve()
        self.sources = {}
        self.missing, self.notes = [], []
        self.cases, self.baselines, self.teachers = [], {}, []
        self.workers = []
        self.application_tests = None
        self.gui_qa = None
        self.full_validation = None
        self.historical = {}
        self.delivery_path = self.inputs/'DELIVERY_MANIFEST.json'
        if not self.delivery_path.is_file(): self.delivery_path=self.inputs/'PREVIEW_DELIVERY_MANIFEST.json'

    def read(self, path, required=False):
        path = Path(path)
        if not path.is_file():
            if required and str(path) not in self.missing: self.missing.append(str(path))
            return None
        data = path.read_bytes()
        value = json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {path}: {value}")))
        self.sources[path.resolve()] = hashlib.sha256(data).hexdigest()
        return value

    def link(self, path, label=None):
        path = Path(path)
        label = label or path.name
        target = os.path.relpath(path, self.output.parent).replace(os.sep, "/")
        return f"[{label}]({target})" if path.is_file() else f"`{target}`（缺失）"

    def from_cloud_path(self, path):
        value = str(path)
        if value.startswith("/content/"): return self.inputs / value.removeprefix("/content/")
        return Path(value) if Path(value).is_absolute() else self.inputs / value

    @staticmethod
    def validate_metrics(row, label, expected_protocol=True):
        for key in ("class_mean_iou", "mean_iou", "mean_boundary_iou"):
            value = row.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f"{label}: {key} must be a ratio in [0,1], not already a percentage")
        per_label = row.get("per_label", {})
        derived = mean([value.get("mean_iou") for value in per_label.values()])
        if derived is not None and row.get("class_mean_iou") is not None and abs(derived-row['class_mean_iou']) > 1e-8:
            raise ValueError(f"{label}: class mIoU does not agree with per-class metrics")
        if expected_protocol:
            for key, expected in (("pairs",59),("views",6),("eval_long_edge",256),("prediction_threshold",.5)):
                if row.get(key) != expected: raise ValueError(f"{label}: unexpected {key}={row.get(key)!r}")
            if len(per_label) != 14: raise ValueError(f"{label}: expected all 14 classes, including failures")
        pair_rows = row.get("per_object_view", [])
        if pair_rows:
            if row.get("pairs",len(pair_rows)) != len(pair_rows): raise ValueError(f"{label}: pair count mismatch")
            for key, target in (("iou","mean_iou"),("boundary_iou","mean_boundary_iou")):
                recomputed = mean([item.get(key) for item in pair_rows])
                if recomputed is not None and row.get(target) is not None and abs(recomputed-row[target]) > 1e-8:
                    raise ValueError(f"{label}: pair mean mismatch: {target}")

    def load(self):
        for family,name,geometry,protocol in CASES:
            folder=self.inputs/family/name
            item={"family":family,"name":name,"geometry":geometry,"protocol":protocol,"folder":folder,
                  "metadata":self.read(folder/'experiment.json',True),"stats":self.read(folder/'training_stats.json',True),
                  "targets":self.read(folder/'targets.json'),"rows":{},"checkpoint_meta":{}}
            for source in (self.inputs/family/'evaluation'/name/'summary.json',
                           self.inputs/'semantic-v2-finalization/closure-evaluation'/name/'summary.json'):
                summary=self.read(source,True)
                if summary:
                    declared=summary.get('experiment',{})
                    if item['metadata'] and declared.get('ply_sha256')!=item['metadata'].get('ply_sha256'):
                        raise ValueError(f"Evaluation/model mismatch: {source}")
                    for row in summary.get('checkpoints',[]):
                        key=(row.get('step'),row.get('representation'))
                        if key[0] not in STEPS or key[1] not in REPRESENTATIONS:
                            raise ValueError(f"Undeclared checkpoint or representation: {source}: {key}")
                        self.validate_metrics(row,f"{name}/{key}")
                        if key in item['rows'] and item['rows'][key]['value']!=row:
                            raise ValueError(f"Conflicting repeated result: {name}/{key}")
                        item['rows'][key]={"value":row,"source":source}
            for step in STEPS:
                item['checkpoint_meta'][step]=self.read(folder/f'step_{step:05d}/checkpoint.json')
                for representation in REPRESENTATIONS:
                    if (step,representation) not in item['rows']:
                        self.missing.append(f"{name}: step={step}, representation={representation}")
                hashes={item['rows'][(step,rep)]['value'].get('probabilities_sha256') for rep in REPRESENTATIONS if (step,rep) in item['rows']}
                hashes.discard(None)
                if len(hashes)>1: raise ValueError(f"{name}/{step}: representations use different probability checkpoints")
            meta,stats=item['metadata'],item['stats']
            if meta:
                if any(meta.get(key) is not True for key in ('geometry_frozen','opacity_frozen')) or meta.get('gt_read_for_training') is not False:
                    raise ValueError(f"{name}: missing fixed-geometry/train-only provenance")
                split=meta.get('split',{})
                if set(meta.get('training_views',[])) & set(split.get('test',[])):
                    raise ValueError(f"{name}: semantic training contains held-out views")
                if stats and stats.get('shape') != [meta.get('gaussian_count'),len(meta.get('classes',[]))]:
                    raise ValueError(f"{name}: training shape differs from model/class metadata")
            self.cases.append(item)
        for geometry in ('balanced','fine'):
            path=self.project/f'benchmarks/results/a100/semantic-results/{geometry}/summary.json'
            data=self.read(path,geometry=='balanced')
            if data:
                self.validate_metrics(data.get('semantics',data),str(path),False)
                self.baselines[geometry]={'source':path,'data':data,'metrics':data.get('semantics',data)}
        textfix=self.inputs/'semantic-v2-results/textfix-only/summary.json'
        data=self.read(textfix,True)
        if data:
            self.validate_metrics(data.get('semantics',data),str(textfix),False)
            self.baselines['textfix-only']={'source':textfix,'data':data,'metrics':data.get('semantics',data)}
        for family,protocol in (('semantic-v2-results','R1'),('semantic-v2b-results','R2')):
            path=self.inputs/family/'masks24/training_masks.json'
            self.teachers.append({'source':path,'protocol':protocol,'data':self.read(path,True)})
            self.read(self.inputs/family/'run-plan.json',True)
        for folder,title in (('semantic-worker-cuda-validation','有先验的 400 步生产验证'),
                             ('semantic-worker-fresh-validation','无先验的 1200 步生产验证')):
            path=self.inputs/folder/'summary.json'
            worker={'folder':folder,'title':title,'path':path,'summary':self.read(path),
                    'failure':self.read(path.with_name('failed.json')),'recovery':self.read(path.with_name('recovery.json')),
                    'invocation':self.read(path.with_name('worker-invocation.json')),'plan':self.read(path.with_name('run-plan.json'))}
            if worker['summary']:
                for row in worker['summary'].get('evaluation',{}).get('summaries',[]):
                    self.validate_metrics(row,folder+'/'+row.get('representation','?'))
            elif not worker['failure']: self.missing.append(str(path))
            self.workers.append(worker)
        delivery=self.read(self.delivery_path)
        self.read(self.inputs/'semantic-v2-finalization/all-checkpoint-inventory.json')
        self.read(self.project/'docs/SEMANTIC_V2_PREVIEW_VALIDATION.json')
        self.application_tests = self.read(self.project/'docs/SEMANTIC_V2_TESTS.json')
        self.gui_qa = self.read(self.project/'docs/SEMANTIC_V2_GUI_QA.json')
        self.full_validation = self.read(self.project/'docs/SEMANTIC_V2_RESULTS_VALIDATION.json')
        for mapping in (self.full_validation or {}).get('destination_mappings', []):
            for key,digest_key in (('extracted_path','archive_sha256'),('preserved_path','preserved_sha256')):
                path=(self.inputs/mapping[key]).resolve()
                if not path.is_relative_to(self.inputs):
                    raise ValueError('Delivery snapshot mapping escapes input directory')
                if self.read(path,True) is not None and self.sources[path]!=mapping[digest_key]:
                    raise ValueError('Delivery snapshot mapping hash mismatch: '+mapping[key])
        historical_paths = {
            'training':'benchmark-results/summary.json', 'rgb':'full-resolution-results/summary.json',
            'priority_comparison':'teatime-priority/comparison.json',
            'priority_roi':'semantic-results/balanced-priority/summary.json',
            'priority_masks':'teatime-priority/priority-mask-provenance.json',
            'priority_patch':'teatime-priority/balanced/priority-patch-evidence.json'}
        for key,relative in historical_paths.items():
            path=self.project/'benchmarks/results/a100'/relative
            self.historical[key]={'path':path,'data':self.read(path,True)}
        if delivery:
            mappings={item['archive_path']:item['extracted_path'] for item in (self.full_validation or {}).get('destination_mappings',[])} if self.delivery_path.name=='DELIVERY_MANIFEST.json' else {}
            declared={mappings.get(item['path'],item['path']):item['sha256'] for item in delivery.get('included_files',[]) if 'path' in item and 'sha256' in item}
            for path,digest in self.sources.items():
                if path.is_relative_to(self.inputs):
                    relative=path.relative_to(self.inputs).as_posix()
                    if relative in declared and declared[relative]!=digest:
                        raise ValueError(f'Delivered JSON hash mismatch: {relative}')
        return self

    def row(self, item, step, representation):
        return item['rows'].get((step,representation),{}).get('value',{})

    def metric(self, name, step=400, representation='editable_binary', key='class_mean_iou'):
        item=next(item for item in self.cases if item['name']==name)
        return self.row(item,step,representation).get(key)

    def plot(self):
        try:
            os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'splat-semantic-report-matplotlib'))
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.notes.append('matplotlib 不可用：未生成曲线，全部数字仍列于表中。')
            return None
        folder=self.output.parent/'semantic-v2-figures';folder.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,3,figsize=(16,4.6),sharey=True)
        styles=['o','s','^','D','v','P']
        for axis,representation in zip(axes,REPRESENTATIONS):
            for item,marker in zip(self.cases,styles):
                values=[(step,self.row(item,step,representation).get('class_mean_iou')) for step in STEPS]
                values=[(step,value) for step,value in values if value is not None]
                if values:axis.plot([x for x,_ in values],[100*y for _,y in values],marker=marker,label=item['name'],linewidth=1.6)
            baseline=self.baselines.get('balanced',{}).get('metrics',{}).get('class_mean_iou')
            if baseline is not None:axis.axhline(100*baseline,linestyle='--',color='#555555',linewidth=1,label='balanced original projection')
            axis.set_title(REP_NAMES[representation]);axis.set_xlabel('Semantic optimizer steps');axis.set_xticks(STEPS)
            axis.grid(alpha=.22);axis.set_ylim(0,100)
        axes[0].set_ylabel('Class mIoU (%)')
        handles,labels=axes[-1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.04))
        fig.suptitle('Teatime: all declared checkpoints; 400-step peaks are post-hoc observations',fontsize=12)
        fig.tight_layout(rect=[0,.13,1,.93])
        path=folder/'all-checkpoint-miou.png';fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
        return path

    def render(self, plot_path=None):
        parts=[]
        add=parts.append
        add('# 3DGS 后置语义优化：完整实验与应用报告')
        add('本报告由本地交付 JSON 自动生成，所有 IoU 从原始 0–1 比例转换为百分比。'+
            '生成时间：'+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'。数据来源根目录：`'+str(self.inputs)+'`。')
        if self.missing:
            add('**当前为未完成的数据报告。以下缺项不以零值或推测补齐：**\n\n'+'\n'.join('- `'+x+'`' for x in self.missing))
        add('## 结果判断与使用建议')
        baseline=self.baselines.get('balanced',{}).get('metrics',{}).get('class_mean_iou')
        simple=self.metric('simple-8');aligned=self.metric('aligned-improved-24')
        fresh=next((record for record in self.workers if record['folder']=='semantic-worker-fresh-validation'),None)
        if fresh and fresh['summary']:
            fresh_rows={row.get('representation'):row for row in fresh['summary'].get('evaluation',{}).get('summaries',[])}
            fresh_binary=fresh_rows.get('raw_binary',{}).get('class_mean_iou');fresh_closed=fresh_rows.get('closed_binary',{}).get('class_mean_iou')
            if fresh_binary is not None:
                difference=f'，相对旧 balanced 原投影变化 {number(100*(fresh_binary-baseline),3)} 个百分点' if baseline is not None else ''
                add(f'**最接近应用纯自动 CUDA 初始化的独立验证：fresh1200 binary 为 {percent(fresh_binary)}%，closed 为 {percent(fresh_closed)}%**{difference}。'
                    '该运行无已有语义先验、无显式层级；仍复用既有完整模型和原8张缓存mask，并非任意新照片的精度承诺。不要用下方有初始化先验的研究检查点最高分替代这个产品入口的实际测量。依据：'+self.link(fresh['path'])+'。')
        if baseline is not None and simple is not None and aligned is not None:
            add(f'旧 balanced 原投影 class mIoU 为 **{percent(baseline)}%**。400 步的 simple-8 可编辑 binary 为 **{percent(simple)}%**，'
                f'R2 aligned-improved-24 binary 为 **{percent(aligned)}%**；后者的 soft 为 **{percent(self.metric("aligned-improved-24",representation="soft"))}%**。'
                f'这两个 400 步 binary 相差 **{number(100*(aligned-simple),3)} 个百分点**，且 R2 使用额外 16 个 teacher 视角与标签对齐。'
                '不能把这个差值全部归于复杂损失，也不能把 soft 分数当成删除操作的 membership 精度。')
        add('400 步是在全部预设结果完成后观察到的检查点，**不是经过独立验证的最佳默认值**。研究采用预先冻结的 0/400/1200/2400 检查点；以下保留全部结果，'
            '不通过 GT 重选阈值、删除失败类别或只展示最高分。当前应用总体默认仍为 `projection`；CUDA 多轮为实验功能，实验性优化内部采用的 `improved + 1200` 是预设，不能宣传为推荐精度配置。'
            '未来若改成 simple + 400，应明确属于看过 Teatime 开发场景后的产品选择，并在独立场景验证。')
        comparison=[]
        for item in self.cases:
            v400=self.row(item,400,'editable_binary').get('class_mean_iou');v1200=self.row(item,1200,'editable_binary').get('class_mean_iou');v2400=self.row(item,2400,'editable_binary').get('class_mean_iou')
            comparison.append([item['name'],percent(v400),percent(v1200),percent(v2400),number(100*(v2400-v400),3) if v2400 is not None and v400 is not None else '—'])
        add(table(['变体','binary 400 (%)','binary 1200 (%)','binary 2400 (%)','2400−400（百分点）'],comparison))
        add('简单 BCE 与复杂组合的结果不支持“损失越多、训练越久就越精确”的保证。有限/错误 teacher 监督的过拟合、类别概率校准与固定采样预算都可能影响结果；'
            '需要单模块消融才能判断具体原因，不能仅凭总 mIoU 将退化唯一归因于某项正则。本轮采用预设seed 0，没有重复种子置信区间，不能将小幅差值表述为统计显著的优越性。')
        if plot_path:add('![全部检查点曲线]('+os.path.relpath(plot_path,self.output.parent).replace(os.sep,'/')+')')
        add('## 原有重建与重要物品能力（历史A100结果，本轮未重训）')
        training=(self.historical['training']['data'] or {}).get('results',[])
        rgb={(row['scene'],row['mode']):row for row in (self.historical['rgb']['data'] or {}).get('results',[])}
        modes={'fast':'快速','balanced':'均衡','fine':'精细'}
        historical_rows=[]
        for scene in ('bonsai','teatime'):
            for mode in modes:
                source=next((row for row in training if row['scene']==scene and row['mode']==mode),None)
                native=rgb.get((scene,mode))
                if source is None or native is None: continue
                timing=source['timing'];quality=native['evaluation']
                historical_rows.append([scene.title(),modes[mode],source['iterations'],
                    '÷'+str(source['resolution_divisor'])+'：'+','.join('×'.join(map(str,size)) for size in timing['train_image_sizes']),
                    number(timing['training_loop_wall_seconds'],3),number(quality['mean_psnr_db'],4),number(quality['mean_ssim'],6)])
        add('以下为此前 NVIDIA A100-SXM4-40GB 上的原始CUDA 3DGS实测，每配置仅seed 0一次运行。快速、均衡、精细同时改变RGB迭代预算、训练尺寸及随训练产生的动态增密；'
            '精细固定22,000步，与本轮400/1200/2400语义优化步数独立。分辨率除数相对下载基底：Bonsai为780×520的images_4，Teatime为988×730，均不是泛指传感器原片。')
        add(table(['场景','模式','RGB步数','训练尺寸除数与像素','训练循环(s)','原尺寸PSNR(dB)','原尺寸SSIM'],historical_rows))
        add('训练秒数取 `training_loop_wall_seconds`，已排除场景初始化与checkpoint保存，不含下载、编译、SfM、语义、评价和应用导出，不能当作上传至成品的总ETA。'
            'RGB质量加载完整高斯、SH3颜色，在各场景下载基底原尺寸评价，Bonsai37张/Teatime27张留出图；'
            '不是直接256px渲染分数，两者排名差异与完整历史结果保留在 '+self.link(self.project/'benchmarks/REPORT.zh-CN.md','原始A100报告')+'。'
            '训练依据 '+self.link(self.historical['training']['path'],'6条训练JSON')+'；原尺寸质量依据 '+self.link(self.historical['rgb']['path'],'7个模型RGB JSON')+'。'
            '作者全场景COLMAP初始化可能利用留出照片，本轮没有重新进行仅训练图SfM、两视图补全或几何误差评估。')
        baseline_roi=self.baselines.get('balanced',{}).get('data',{}).get('rgb_regions',{}).get('priority_roi',{})
        priority_roi=(self.historical['priority_roi']['data'] or {}).get('rgb_regions',{}).get('priority_roi',{})
        priority_timing=self.historical['priority_comparison']['data'] or {}
        if baseline_roi and priority_roi and priority_timing:
            prior_psnr=baseline_roi['mean_psnr_db'];priority_psnr=priority_roi['mean_psnr_db']
            prior_seconds=priority_timing['baseline_training_loop_seconds'];priority_seconds=priority_timing['priority_training_loop_seconds']
            add('重要物品功能已有独立历史对照：Teatime均衡保持15,000步、494×365、seed 0与相同划分，'
                '使用8张训练图预测熊mask，将RGB L1的目标像素权重设为3、其余为1并归一化，保留SSIM、优化器和动态增密；人工GT仅用于评价。'
                f'同一 {cell(baseline_roi.get("views"))} 张256px留出图的熊ROI PSNR由 **{number(prior_psnr,4)}→{number(priority_psnr,4)} dB**，'
                f'变化 **+{number(priority_psnr-prior_psnr,4)} dB**；训练循环由 {number(prior_seconds,3)}→{number(priority_seconds,3)}秒，'
                f'增加 **{number(priority_seconds-prior_seconds,3)}秒**。'
                '证据：'+self.link(self.baselines['balanced']['source'],'原均衡ROI')+'、'+self.link(self.historical['priority_roi']['path'],'重点熊ROI')+'、'+
                self.link(self.historical['priority_comparison']['path'],'同预算训练对照')+'。')
        add('此前独立benchmark按相机名称stem写入并读取同名PNG，mask命名原本已正确匹配；见 '+
            self.link(self.historical['priority_masks']['path'],'历史mask来源')+'、'+self.link(self.historical['priority_patch']['path'],'历史损失修改证据')+'、'+
            self.link(self.project/'benchmarks/results/a100/teatime-priority/balanced/priority_loss_hook.py','实际历史读取hook')+'。'
            '上述ROI收益不是本轮应用mask命名修复后的端到端再验收，也不证明其他物品、其他场景或每个区域均改善；本轮没有重训RGB。'
            '学习式极少视角补全、RGB与语义联合训练、可靠实例与完整多层级语义、原生Windows运行验收仍未完成，见 '+
            self.link(self.project/'docs/SEMANTIC_V2_LIMITATIONS.zh-CN.md','当前实现局限')+' 和原始报告。')
        add('## 固定评价协议与原始基线')
        add('评价使用 Teatime 的 14 类、59 个物品—视角对、6 个留出视角、256 像素最长边、固定 0.5 阈值及 2 像素内侧 Boundary IoU。'
            'class mIoU 先按类跨视角求均值，再等权平均14类；pair IoU 与边界均值按59对平均。触边分组由 GT 是否碰到图片边缘决定，属于评价切片。'
            '所有分数都是二维投影类别掩码指标，没有三维或实例 GT。SfM/相机来源仍是已有全场景 COLMAP，须保留这一传导式几何来源限制。')
        baseline_rows=[]
        for key,item in self.baselines.items():
            m=item['metrics'];baseline_rows.append([key,percent(m.get('class_mean_iou')),percent(m.get('mean_iou')),percent(m.get('mean_boundary_iou')),number(item['data'].get('fusion_seconds')),self.link(item['source'])])
        add(table(['基线/对照','class mIoU (%)','pair IoU (%)','Boundary (%)','融合时间(s)','依据'],baseline_rows))
        add('`textfix-only` 仅做声明的文字规范化对照，不改变缓存掩码像素；它不是 R2 token 对齐重推理。step 0 是各实验的概率初始化与分类目录，不直接冒充原始硬融合基线。'
            'fine 使用既有 22,000 步 RGB 几何，balanced 使用既有 15,000 步；本轮语义均冻结原高斯 xyz、scale、rotation、opacity 与 SH/RGB，不重训 RGB。')
        add('## 全部变体、检查点与三种表示')
        add('soft：渲染原始语义概率后以0.5阈值判断；binary：先对每高斯概率阈值化再渲染；closed：先将子类成员合入祖先再渲染。'
            '闭包是派生编辑规则，不能与原始概率或原始 membership 混为同一测量。下表全部数值为百分比；“触边/图内”单元为 pair IoU / Boundary IoU。')
        for item in self.cases:
            add('### '+item['name'])
            add('依据：'+self.link(item['folder']/'experiment.json','experiment.json')+'；'+
                self.link(self.inputs/item['family']/'evaluation'/item['name']/'summary.json','soft/binary 全量评价')+'；'+
                self.link(self.inputs/'semantic-v2-finalization/closure-evaluation'/item['name']/'summary.json','closed 全量评价')+'。')
            rows=[]
            for step in STEPS:
                for rep in REPRESENTATIONS:
                    r=self.row(item,step,rep);groups=r.get('image_border_groups',{});border=groups.get('touching_image_border',{});inside=groups.get('inside_image',{})
                    rows.append([step,REP_NAMES[rep],percent(r.get('class_mean_iou')),percent(r.get('mean_iou')),percent(r.get('mean_boundary_iou')),
                        percent(border.get('mean_iou'))+' / '+percent(border.get('mean_boundary_iou')),
                        percent(inside.get('mean_iou'))+' / '+percent(inside.get('mean_boundary_iou')),
                        str(border.get('pairs','—'))+' / '+str(inside.get('pairs','—'))])
            add(table(['步数','表示','class mIoU','pair IoU','Boundary','触边 IoU/边界','图内 IoU/边界','触边/图内对数'],rows))
        add('## 逐类：400 步与最终 2400 步')
        add('每个单元为 **类别 IoU / 类别 Boundary IoU（%）**，保留所有零分类。400步列用于完整解释已观察到的早期峰值，最终列固定为2400步；不按分数改称“最终最佳”。'
            '逐视角原始记录在各 summary 的 `per_object_view`，可用于定位触边、漏检和小部件失败。')
        for item in self.cases:
            add('### '+item['name'])
            labels=sorted({label for entry in item['rows'].values() for label in entry['value'].get('per_label',{})})
            rows=[]
            for label in labels:
                values=[]
                for rep in REPRESENTATIONS:
                    for step in (400,2400):
                        metric=self.row(item,step,rep).get('per_label',{}).get(label,{})
                        values.append(percent(metric.get('mean_iou'))+' / '+percent(metric.get('mean_boundary_iou')))
                rows.append([label,*values])
            add(table(['类别','soft 400','soft 2400','binary 400','binary 2400','closed 400','closed 2400'],rows))
        add('## 输入、训练预算与时间成本')
        setup=[]
        timing=[]
        for item in self.cases:
            meta=item['metadata'] or {};stats=item['stats'] or {};cfg=stats.get('config',{})
            setup.append([item['name'],item['geometry'],meta.get('gaussian_count','—'),len(meta.get('training_views',[])),meta.get('training_mask_count','—'),len(meta.get('classes',[])),
                len(stats.get('active_classes',[])),meta.get('preset','—'),meta.get('model','—')])
            segments=[stats.get(key) for key in ('setup_seconds','training_seconds','callback_seconds')]
            subtotal=sum(segments) if all(isinstance(v,(float,int)) for v in segments) and not stats.get('resumed_from_step') else None
            timing.append([item['name'],number(segments[0]),number(segments[1]),number(segments[2]),number(subtotal),
                           gib(stats.get('cuda_peak_allocated_bytes')),gib(stats.get('cuda_peak_reserved_bytes')),stats.get('resumed_from_step','—')])
        add(table(['变体','RGB模型','完整高斯数','训练视角','mask数','训练类别数','有效类别数','preset','GPU'],setup))
        configs=[]
        for item in self.cases:
            meta=item['metadata'] or {};cfg=(item['stats'] or {}).get('config',{})
            configs.append([item['name'],meta.get('train_size','—'),cfg.get('learning_rate','—'),meta.get('seed','—'),cfg.get('balanced_bce','—'),
                cfg.get('dice_weight','—'),str(cfg.get('boundary_radius','—'))+' / '+str(cfg.get('boundary_weight','—')),
                cfg.get('hierarchy_weight','—'),cfg.get('prior_weight','—')])
        add(table(['变体','训练最长边','学习率','seed','均衡BCE','Dice权重','边界半径/权重','层级权重','先验权重'],configs))
        add(table(['变体','setup(s)','optimizer(s)','callback(s)','三段合计(s)*','峰值allocated GiB','峰值reserved GiB','恢复起点'],timing))
        add('`setup_seconds` 是语义优化函数内的参数/观测准备；不包含调用前的完整 PLY 加载、目标构建等外层准备。`training_seconds` 按源码为 optimizer 循环，包含渲染与日志归约、排除 setup 与 callback。'
            '`callback_seconds` 包括检查点、恢复状态保存与回调。*三段合计仅为已记录片段之和，不是完整进程 wall，更不是包含 teacher 的端到端耗时；发生恢复时不能将累计 optimizer 与本段 setup/callback 当完整总时长。'
            '研究进程未单独提供 wall 字段时本报告不推算。CUDA 峰值为 PyTorch allocator 统计，不等于整卡所有进程用量。')
        visit_rows=[]
        for item in self.cases:
            stats=item['stats'] or {};meta=item['metadata'] or {};counts=stats.get('class_visits',[]);views=list(stats.get('view_visits',{}).values())
            visit_rows.append([item['name'],stats.get('completed_steps','—'),stats.get('config',{}).get('channels_per_step','—'),
                f'{min(counts)}–{max(counts)}' if counts else '—',f'{min(views)}–{max(views)}' if views else '—',self.link(item['folder']/'training_stats.json')])
        add(table(['变体','实际更新步数','每步最多类别通道','各类更新次数范围','各视角访问次数范围','完整loss/访问依据'],visit_rows))
        add('一步是一次优化器更新，不是所有类别与视角的一整轮。增加视角或异常类别，会改变固定2400步下的监督分配；不能仅以“轮数相同”声称训练曝光相同。')
        add('### 二维 teacher 与 R2 标签对齐')
        teacher_rows=[]
        for record in self.teachers:
            data=record['data'] or {};masks=data.get('masks',[]);diag=data.get('label_alignment',[])
            sums={key:sum(int(item.get(key,0)) for item in diag if isinstance(item,dict)) for key in ('detected_boxes','accepted_labels','rejected_ambiguous')}
            teacher_rows.append([record['protocol'],len(data.get('old_views',[])),len(data.get('new_views',[])),data.get('old_count','—'),
                len(masks)-data['old_count'] if 'old_count' in data else '—',len(masks) if data else '—',number(data.get('seconds')),
                str(data.get('config',{}).get('strict_query_labels','未记录')),sums['accepted_labels'] if diag else '—',sums['rejected_ambiguous'] if diag else '—',self.link(record['source'])])
        add(table(['阶段','旧视角','新增视角','旧mask','新增mask','总mask','记录teacher(s)','strict对齐','接受框','歧义拒绝框','依据'],teacher_rows))
        add('teacher 秒数从准备脚本中模型缓存路径解析之后开始，包含逐图调用、推理和新mask保存，不包含此前模型下载；逐图调用可能包含模型重复装载，不能视为纯神经网络 kernel 时间。'
            '原8张缓存的历史成本未重计为本轮新增成本。R1 与 R2 的新增16图相同，R2 按查询词 token 范围对齐，最高均分阈值0.20、领先第二名至少0.025；低置信或歧义框拒绝。'
            '两组都保留原8张缓存像素；R2并非清洗全部历史mask，也不会凭名称补造缺失目标。更多视角是额外teacher信息，时间和精度必须单列。')
        add('## 层级、未知区域与编辑语义')
        hierarchy=[]
        for item in self.cases:
            meta=item['metadata'] or {};classes=meta.get('classes',[]);edges=meta.get('hierarchy_edges',[])
            edge_names=[classes[p]+' ← '+classes[c] for p,c in edges if 0<=p<len(classes) and 0<=c<len(classes)]
            hierarchy.append([item['name'],'；'.join(edge_names) or '无',number((item['stats'] or {}).get('config',{}).get('hierarchy_weight'),3),self.link(item['folder']/'targets.json','全部候选/拒绝诊断')])
        add(table(['变体','保留的类别关系（父←子）','层级loss权重','依据'],hierarchy))
        add('关系先验来自运行前的查询语义声明，再要求多视角掩码支持；单纯几何包含发现过错误的 pouf→cookies 候选，不能自动当成物理部件。'
            'simple 的层级loss权重为0，即使另行提供相同的编辑闭包，也不能称为训练了层级损失。无支持的 bear nose、hooves/sheep 等边必须保留拒绝原因；“未建边”不是“无需报告”。')
        diagnostics=[]
        for item in self.cases:
            for step in STEPS:
                diag=self.row(item,step,'hierarchy_closed_binary').get('membership_diagnostics',{})
                diagnostics.append([item['name'],step,percent(diag.get('raw_unknown_fraction')),percent(diag.get('effective_unknown_fraction')),
                    percent(diag.get('raw_multilabel_fraction')),percent(diag.get('effective_multilabel_fraction')),
                    diag.get('raw_hierarchy_violations','—'),diag.get('effective_hierarchy_violations','—')])
        add(table(['变体','步数','raw未知(%)','closed未知(%)','raw多标签(%)','closed多标签(%)','raw违反数','closed违反数'],diagnostics))
        add('这些比例覆盖完整PLY，包括不可见/遮挡高斯，不是GT准确率。闭包会将子类成员合入祖先，强制消除包含违反并不能证明预测正确。'
            '当前是同标签 category union，不是实例 ID；多标签和层级编辑只实现了部分目标，尺度条件latent尚未实现。')
        add('## 独立生产 CUDA worker 验证')
        add('两个生产验证均使用独立 worker 调用及完整 balanced 模型/原8图，不并入冻结 R1/R2 表。seeded400 使用身份验证的step-zero先验和显式层级；'
            'fresh1200 使用均匀0.05、无先验且 approved=[]，更接近应用纯自动迭代入口。前者不能替代后者的品质证据；两者都不是从照片上传开始的端到端桌面重建验证。')
        for record in self.workers:
            add('### '+record['title'])
            add('依据：'+self.link(record['path'])+'；'+self.link(record['path'].with_name('run-plan.json'),'运行声明')+'。')
            if record['failure']:
                failure=record['failure']
                add('**保留最初失败记录：** '+cell(failure.get('error_type',''))+' '+cell(failure.get('message',''))+'。依据：'+self.link(record['path'].with_name('failed.json'))+'。后续成功不会覆盖此失败。')
            if record['recovery']:
                add('后续恢复以 '+self.link(record['path'].with_name('recovery.json'))+' 为准；评价阶段重新加载模块失败与语义训练失败应分开，不能把评价恢复算成重新训练。')
                attempts=record['recovery'].get('attempts',[])
                add(table(['恢复尝试','状态','重训','原训练步数','恢复(s)','原文件保留','跨阶段修改的源码'],[[a.get('attempt','—'),a.get('status','—'),a.get('training_repeated','—'),
                    a.get('original_semantic_steps','—'),number(a.get('recovery_seconds')),a.get('original_files_preserved','—'),'；'.join(a.get('code_deviations',{})) or '无记录'] for a in attempts]))
                for attempt in attempts:
                    reentry=attempt.get('loader_reentry_validation',{})
                    if reentry:add('恢复验证记录：同进程连续加载 '+str(reentry.get('consecutive_same_process_loads','—'))+' 次，冻结张量哈希一致='+str(reentry.get('tensor_hashes_identical','未记录'))+'。')
                add('恢复修复了合法 namespace package 的同进程再次加载检查，使用已有 worker 导出继续评价，并记录重新加载验证与跨阶段源码差异；不能声称失败不存在或代码全程未变。')
            if record['summary']:
                w=record['summary'];rows=w.get('evaluation',{}).get('summaries',[])
                plan=w.get('run_plan') or record.get('plan') or {};cfg=plan.get('configuration',{});context=plan.get('app_usage_context',{})
                add(table(['优化步数','preset','初始化','批准关系数','训练视角/mask'],[[cfg.get('semantic_steps','—'),cfg.get('preset','—'),
                    context.get('initialization','已有验证先验' if cfg.get('prior') else '未提供先验'),len(cfg.get('approved_hierarchy',[])),
                    str(len(plan.get('training_views',[])))+' / '+str(plan.get('training_mask_count','—'))]]))
                add(table(['状态','原始输入未变','代码哈希未变','worker总(s)','optimizer(s)','eval(s)','整个验证run(s)'],[[w.get('status','—'),w.get('all_original_inputs_unchanged','—'),
                    w.get('code_hashes_unchanged','—'),number(w.get('worker_total_seconds')),number(w.get('worker_optimizer_seconds')),number(w.get('evaluation_seconds')),number(w.get('full_run_seconds'))]]))
                add(table(['生产表示','class mIoU(%)','pair IoU(%)','Boundary(%)'],[[r.get('representation','—'),percent(r.get('class_mean_iou')),percent(r.get('mean_iou')),percent(r.get('mean_boundary_iou'))] for r in rows]))
                by_rep={row.get('representation'):row for row in rows}
                raw=by_rep.get('raw_binary',{}).get('class_mean_iou');closed=by_rep.get('closed_binary',{}).get('class_mean_iou')
                if raw is not None and closed is not None:
                    add(f'本运行 raw binary → closed binary：{percent(raw)}% → {percent(closed)}%，变化 **{number(100*(closed-raw),3)} 个百分点**。'
                        +('`contents_of` 的 coffee→coffee mug 祖先闭包会把杯内内容也纳入杯子选择集合，可能与仅标杯体的 coffee mug GT 定义不同。'
                          '因此“选择父类别时包含内容”是编辑语义的改变，不能把闭包违反数为零解释为精度提升；负收益保留。'
                          if cfg.get('approved_hierarchy') else '本次没有批准层级关系，闭包不新增祖先成员；两种表示相同不能解释为自动发现了可靠层级。'))
                scopes={key:w[key] for key in w if 'timing_scope' in key or 'recovery' in key or 'retrain' in key}
                if scopes:add('本次计时/恢复原字段：`'+cell(json.dumps(scopes,ensure_ascii=False))+'`。')
            elif record['invocation']:
                call=record['invocation']
                add('已记录 worker 调用结果：完成='+str(call.get('worker_complete','未记录'))+'，调用耗时='+number(call.get('worker_total_seconds'))+'秒。最终评价未完成，不报告分割精度。')
            else:add('没有可核验的完成结果，不声明通过，缺失指标不填0。')
        add('生产 `worker_total_seconds` 覆盖整个公开 worker 调用，包括模型加载、身份/标签处理、训练、检查点及导出；完整新运行的 `full_run_seconds` 覆盖验证 run 的准备、worker、评价和导出，排除此前 Python 模块导入。'
            '如发生仅评价恢复，完整连续 run 墙钟可能不再可得，保留原字段及恢复说明，不将多个片段拼成一次端到端计时。与研究函数的 optimizer 秒数也不能直接互换。')
        add('## 应用接入、论文借鉴与未完成能力')
        add('应用支持新重建后选择 CUDA 多轮语义，也支持项目内已有完整运行的 semantic-only 新任务，共用单GPU队列并传播取消与失败。'
            '输入必须是本项目已完成的运行、完整PLY与原照片/相机/mask；外部独立导入的JSON/PLY不能冒充可续训项目。MPS/CPU仅走原投影回退，不能声明运行了CUDA语义优化。'
            '原PLY顺序与哈希保持，概率/二值/闭包另存，预览JSON不是完整高阶SH渲染。')
        add('手动 parent_id 可转换为显式标签关系，来源写为 user_manual_parent_id；直接API声明为 user_explicit，自动几何包含不会获得批准。'
            '新增本地检测启用严格查询对齐。重点物品RGB mask命名已与上游读取路径对齐，但这一修复不等于本轮新做了重点几何训练消融。'
            '联合RGB+语义训练仍未实现；语义22k不是用户要求，22k仅是RGB fine总步数。')
        if self.application_tests:
            tests = self.application_tests
            py, js = tests.get('python', {}), tests.get('javascript', {})
            add('应用最终回归证据：'+self.link(self.project/'docs/SEMANTIC_V2_TESTS.json','SEMANTIC_V2_TESTS.json')+
                f'（应用 {cell(tests.get("application_version"))}，打包修订 {cell(tests.get("packaged_revision"))}，完成时间 {cell(tests.get("completed_utc"))}）。'
                f'Python **{cell(py.get("passed"))} 项通过，另有 {cell(py.get("subtests_passed"))} 个子测试通过**，耗时 {number(py.get("seconds"))} 秒；'
                f'JavaScript **{cell(js.get("passed"))} 项通过**，耗时 {number(js.get("seconds"))} 秒。'
                '这些是CPU/API/算法/sidecar及前端行为回归，不是新增A100精度评价，也不是MPS多轮训练证明。')
            bridge = tests.get('loopback_result_bridge', {})
            if bridge.get('rerun_in_this_pass') is False:
                add(f'另存的 {cell(bridge.get("passed"))} 项回传桥检查本次没有重跑，未并入上述Python或JavaScript通过数。')
            guards = tests.get('coordinate_grid_guards', {})
            if guards.get('added_after_research_freeze'):
                add('最终产品坐标保护修改晚于冻结的A100研究：默认投影跳过明确标记为未核验的相机；重点物品RGB训练拒绝未核验的原图/注册图坐标网格，以及尺寸不符的整幅mask。'
                    f'新增 {cell(guards.get("new_regression_cases"))} 项坐标保护回归。研究指标是否因此重算：{cell(guards.get("benchmark_results_recomputed"))}；'
                    '这些保护不能追溯改写原研究代码、旧baseline或已记录的A100结果。')
                add(table(['最终坐标保护源码','测试记录中的SHA256'],[
                    [self.link(self.project/path,path),digest] for path,digest in guards.get('source_sha256',{}).items()
                ]))
                worker_path = self.project/'backend/semantic_worker.py'
                current_worker = hashlib.sha256(worker_path.read_bytes()).hexdigest() if worker_path.is_file() else None
                fresh_record = next((record for record in self.workers if record['folder']=='semantic-worker-fresh-validation'),{})
                fresh_worker = (fresh_record.get('plan') or {}).get('code_sha256',{}).get('backend/semantic_worker.py')
                if current_worker and fresh_worker:
                    add('生产CUDA语义worker与独立fresh A100验证时的源码哈希相同：'+cell(current_worker==fresh_worker)+
                        '，当前 SHA256=`'+current_worker+'`；依据 '+self.link(self.inputs/'semantic-worker-fresh-validation/run-plan.json','fresh运行计划')+'。'
                        '这项比较只证明worker文件是否变化，不把后续semantics/upstream坐标保护视为已经重新进行A100验证。')
        if self.gui_qa:
            gui = self.gui_qa
            preview, application = gui.get('preview', {}), gui.get('application', {})
            add('真实预览界面验收另见 '+self.link(self.project/'docs/SEMANTIC_V2_GUI_QA.zh-CN.md','GUI观察、截图与限制')+' 和 '+
                self.link(self.project/'docs/SEMANTIC_V2_GUI_QA.json','GUI机器可读记录')+'。'
                f'实际范围是 Mac {cell(application.get("package_revision"))} 冻结后端（端口 {cell(application.get("backend_port"))}）配合浏览器WebUI，'
                f'导入 {cell(preview.get("gaussian_count"))} 点、SH{cell(preview.get("sh_degree"))} 抽样预览；'
                f'来源完整模型的高斯数为 {cell(preview.get("source_gaussian_count_from_metadata"))}。'
                '这是交互验收，未验证原生Electron整壳或Windows，也不能替代完整模型的渲染画质与语义指标。')
            check_names = {
                'import':'真实JSON导入', 'search_focus':'类别搜索与定位',
                'hide_bear_undo':'隐藏bear与撤销', 'isolate_bear_undo':'隔离bear与撤销',
                'hide_parent_undo':'隐藏父类别与撤销', 'hide_child_undo':'隐藏子类别与撤销',
                'orbit_zoom':'鼠标旋转与缩放', 'cuda_controls':'MPS上禁用CUDA及外部JSON续训',
                'quality':'抽样预览稀疏孔洞', 'local_text_command':'本地文本指令',
                'root_hide_undo':'场景根节点隐藏与撤销', 'export_reimport':'导出后重导入',
                'external_llm':'外部LLM', 'native_electron_ui':'原生Electron界面',
                'native_windows':'原生Windows', 'full_1706789_interaction':'完整高斯模型交互',
                '100000_preview_interaction_this_round':'本轮100k预览交互', 'sustained_performance':'持续性能'}
            status_names = {'passed':'已观察通过', 'observed_limitation':'已观察限制',
                            'pending_observation':'待实际观察', 'not_tested':'未测试'}
            add(table(['界面项目','记录状态'],[
                [check_names.get(check.get('id'),check.get('id')), status_names.get(check.get('status'),check.get('status'))]
                for check in gui.get('checks', [])]))
            add('该预览来自旧8视角mask及已有语义先验的生产400步运行，仍保留 `##l-e` 等旧teacher标签碎片；'
                '它不是R2严格标签对齐结果，也不是fresh1200结果。类别目录是category union，截图中的稀疏孔洞不能代表完整SH3画质；'
                '可见点数和旋转缩放的短暂观察不能当作mIoU或持续流畅度指标。MPS标识与CUDA控件禁用也不表示在GUI执行了语义训练。')
            timing = gui.get('timing', {})
            if timing.get('file_selection_tool_block_seconds') is not None:
                add(f'GUI文件选择工具曾异常阻塞 {number(timing.get("file_selection_tool_block_seconds"),1)} 秒；该时间不属于应用JSON解析耗时或模型训练时间。'
                    '本次应用导入耗时未得到可靠测量，保留空值。')
        add('SAGA 的作者 v2 代码学习32维尺度门控亲和特征、三维尺度及像素对对比关系：'+
            '[训练代码](https://github.com/Jumpat/SegAnyGAussians/blob/v2/train_contrastive_feature.py)、[尺度计算](https://github.com/Jumpat/SegAnyGAussians/blob/v2/get_scale.py)。'
            'LaGa 使用分层亲和特征与跨层约束，并在推理阶段关联掩码、构建多个CLIP语义原型：'+
            '[亲和训练](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/train_affinity_features.py)、[推理代码](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/inference.ipynb)。'
            '当前没有复现这些尺度 latent、完整实例关联与CLIP多原型，不能将其论文成绩或速度写成本项目结果。')
        add(table(['论文机制','作者源码依据','本轮实际实现','尚未实现/不能声称'],[
            ['SAGA：32维亲和特征、三维mask尺度和尺度门',
             '[v2 train_contrastive_feature.py](https://github.com/Jumpat/SegAnyGAussians/blob/v2/train_contrastive_feature.py) / [get_scale.py](https://github.com/Jumpat/SegAnyGAussians/blob/v2/get_scale.py)',
             '完整高斯 N×C 独立类别概率，每次取至多三类通过RGB rasterizer反传',
             '没有三维尺度提取、Linear(1,32)+Sigmoid尺度门或连续粒度latent'],
            ['SAGA：尺度条件正负像素对、困难关系、局部特征平滑',
             '[训练](https://github.com/Jumpat/SegAnyGAussians/blob/v2/train_contrastive_feature.py) / [特征模型](https://github.com/Jumpat/SegAnyGAussians/blob/v2/scene/gaussian_model_ff.py)',
             '多视角teacher mask共同拟合概率；边界权重与可选包含正则',
             '不是亲和对比学习，也没有复现其kNN特征平滑/困难像素对机制'],
            ['LaGa：分层[16,8,8]亲和特征、区域原型余弦关系、跨层正负约束',
             '[train_affinity_features.py](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/train_affinity_features.py)',
             '类别可重叠；明确批准且经mask支持的父子概率惩罚和编辑闭包',
             '没有对应分层latent、原型亲和训练或完整物体—部件—子部件特征层'],
            ['LaGa：跨视角分组、HDBSCAN、动态K-Means与多个CLIP语义原型',
             '[inference.ipynb](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/inference.ipynb) / [preprocess.py](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/preprocess.py)',
             '已知查询类别按标签取并集，复用一组三维参数',
             '没有可靠实例关联、多CLIP原型或未训练新词的语言查询表示'],
            ['LaGa特征训练可调整opacity；本轮要求语义独立消融',
             '[gaussian_model_ff.py](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/scene/gaussian_model_ff.py)',
             '位置、尺度、旋转、opacity与SH均冻结，仅语义参数更新并验证PLY/张量身份',
             '不能把改变几何或透明度的其他方法结果直接当作本轮同条件收益'],
        ]))
        add('当前工程对照入口：'+self.link(self.project/'backend/semantic_refinement.py','语义概率优化/损失')+'、'+
            self.link(self.project/'backend/semantic_targets.py','类别并集与包含候选')+'、'+self.link(self.project/'backend/semantic_worker.py','CUDA worker与批准关系')+'、'+
            self.link(self.project/'backend/semantic_service.py','项目内续训与模型绑定')+'。这些链接指向当前工程；冻结研究代码版本仍以各 experiment.json 内 code_sha256 为准。')
        add('共享一组三维参数只提供共同表示，不是独立测得的跨视角身份一致性。当前还缺共同可见表面/实例对应的独立一致性指标；'
            '类别并集、显式关系和闭包也不等于可靠实例/多级部件GT。固定几何的跨边界大高斯与未重建细节无法仅用语义概率修复。详见 '+
            self.link(self.project/'docs/SEMANTIC_V2_LIMITATIONS.zh-CN.md','实现局限')+' 和 '+self.link(self.project/'docs/SEMANTIC_RESEARCH_V2.zh-CN.md','论文/源码核对')+'。')
        add('## 研究诚信、交付范围与依据索引')
        add('R1/R2规则按训练监督诊断预先声明，未用本轮GT像素训练或选阈值；但 Teatime 的旧结果已经被查看，本轮并非完全未接触的盲测。'
            'R2修订依据新增训练日志的拼接标签问题，修改teacher标签对齐，不能把R1覆盖成修正版。400步峰值是事后观察，不回写为预注册最优。'
            '默认1200仍为实验预设而非推荐精度；新产品默认如依据本场景调整，必须注明开发集选择并补独立数据。')
        add('运行前依据：'+self.link(self.project/'docs/SEMANTIC_V2_PROTOCOL.zh-CN.md','原协议/R1')+'；'+
            self.link(self.project/'docs/SEMANTIC_V2_PROTOCOL_R2.zh-CN.md','R2协议')+'。')
        manifest=self.read(self.delivery_path) or {}
        inventory=self.inputs/'semantic-v2-finalization/all-checkpoint-inventory.json'
        if manifest:add('当前交付声明：'+cell(manifest.get('selection',manifest.get('scope','未记录')))+'。中间浮点检查点说明：'+cell(manifest.get('intermediate_float_checkpoints','本manifest未声明，按检查点清单与本地存在性区分'))+'。')
        matrices=self.read(inventory) or []
        included=[item for item in matrices if isinstance(item,dict) and item.get('included') is True and 'path' in item]
        excluded=[item for item in matrices if isinstance(item,dict) and item.get('included') is False and 'path' in item]
        local_included=sum((self.inputs/item['path']).is_file() for item in included)
        local_excluded=sum((self.inputs/item['path']).is_file() for item in excluded)
        add(f'检查点清单登记 {len(matrices)} 个 NPZ 文件，其中 **{len(included)} 个纳入完整权重归档交付**（included=true），'
            f'**{len(excluded)} 个未收入该归档**（included=false，逐项见清单）。报告生成时，应交付NPZ本地存在 **{local_included}/{len(included)} 个**；'
            f'未收入归档的NPZ本地存在 {local_excluded}/{len(excluded)} 个。未收入的文件不应被描述为尚欠交付文件，清单登记总数也不等于归档交付数量。'
            '全部72条研究评价可以独立从JSON核验；文件存在性不代替完整归档的哈希与逐项验收，回传完成状态以验收记录为准。'
            '清单中的 included 标志描述原始完整权重归档选择，不代替本地文件存在性。详见 '+self.link(inventory,'检查点清单')+' 和 '+
            self.link(self.delivery_path,'当前交付manifest')+'；预览包验收见 '+self.link(self.project/'docs/SEMANTIC_V2_PREVIEW_VALIDATION.json','验收记录')+'。'
            '完整RGB PLY沿用此前A100成果，不因为本报告自动复制或重新训练。')
        if self.full_validation:
            validation=self.full_validation
            add('完整结果归档验收：'+self.link(self.project/'docs/SEMANTIC_V2_RESULTS_VALIDATION.json','逐文件验收记录')+'。'
                f'记录状态为 {cell(validation.get("status"))}，实际 {cell(validation.get("actual_bytes"))} 字节，SHA256=`{cell(validation.get("actual_sha256"))}`；'
                f'已核验 {cell(validation.get("checked_manifest_files"))} 个清单文件，另含原始manifest。'
                f'ZIP CRC通过={cell(validation.get("zip_crc_verified"))}，清单大小/哈希通过={cell(validation.get("manifest_file_sizes_and_sha256_verified"))}，'
                f'最终目标文件核验数={cell(validation.get("final_destination_files_verified"))}。这些是归档完整性证据，不是新增模型训练或画质验收。')
            for mapping in validation.get('destination_mappings',[]):
                add('归档存在不同时间的状态快照：完整ZIP原路径 `'+cell(mapping['archive_path'])+'` 的 '+cell(mapping.get('archive_phase'))+
                    ' 状态原字节另存为 '+self.link(self.inputs/mapping['extracted_path'],mapping['extracted_path'])+
                    '；后打预览包的 '+cell(mapping.get('preserved_phase'))+' 状态仍保留在 '+
                    self.link(self.inputs/mapping['preserved_path'],mapping['preserved_path'])+
                    '。生成器按验收映射核对两份独立SHA，不将预览的complete快照用于匹配完整ZIP较早快照；原始完整manifest未改写。')
        if self.notes:add('生成说明：'+'；'.join(self.notes))
        add(table(['读取的依据文件','SHA256'],[[self.link(path,os.path.relpath(path,self.inputs) if path.is_relative_to(self.inputs) else path.name),digest] for path,digest in sorted(self.sources.items())]))
        return '\n\n'.join(parts)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=PROJECT/'benchmarks/results/semantic-v2')
    parser.add_argument('--output',type=Path,default=PROJECT/'benchmarks/SEMANTIC_V2_REPORT.zh-CN.md')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--skip-plot',action='store_true')
    args=parser.parse_args(argv)
    report=Report(args.input,args.output).load()
    if report.missing and not args.allow_partial:
        print(json.dumps({'status':'waiting_for_complete_inputs','missing':report.missing},ensure_ascii=False,indent=2))
        return 2
    if args.check_only:
        print(json.dumps({'status':'validated','cases':len(report.cases),'metric_rows':sum(len(item['rows']) for item in report.cases),'missing':report.missing},ensure_ascii=False))
        return 0
    report.output.parent.mkdir(parents=True,exist_ok=True)
    figure=None if args.skip_plot else report.plot()
    body=report.render(figure)
    temporary=report.output.with_suffix(report.output.suffix+'.tmp');temporary.write_text(body,encoding='utf-8');temporary.replace(report.output)
    print(json.dumps({'status':'written','output':str(report.output),'characters':len(body),'metric_rows':sum(len(item['rows']) for item in report.cases),
                      'source_files':len(report.sources),'figure':str(figure) if figure else None,'missing':report.missing},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
