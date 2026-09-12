"""Create a concise report from actual saved MPS benchmark results."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
data = {name: json.loads((RESULTS/name/"summary.json").read_text()) for name in ["bonsai","teatime"]}
lines = ["# 两个真实数据集的 MPS 预览后端实测", "",
         "当前预览器能完成训练和重点区域优化，但画质与语义精度仍不足以作为高质量重建产品交付。本报告保留低分、精细模式退化与局部增强的全局代价。", "",
         "两个场景均为 16 张训练图像、8 张留出测试图像；统一在 128 像素长边评价。使用预计算的全场景 COLMAP 相机与稀疏几何；颜色只从训练图像初始化。结果不代表原始 CUDA 3DGS。", "",
         "| 场景 | 模式 | 测试 PSNR↑ | 测试 SSIM↑ | 优化耗时 |", "|---|---|---:|---:|---:|"]
for name, summary in data.items():
    for run in summary["runs"]:
        lines.append(f"| {name} | {run['name']} | {run['rgb']['mean_psnr_db']:.3f} dB | {run['rgb']['mean_ssim']:.4f} | {run['training_wall_seconds']:.2f} 秒 |")
tea = data["teatime"]
base = next(r for r in tea["runs"] if r["name"]=="balanced")
priority = next(r for r in tea["runs"] if r["name"]=="balanced-priority")
lines += ["", "快速 / 均衡 / 精细分别使用 48 / 80 / 120 步、96 / 160 / 224 个高斯、64 / 96 / 128 像素训练长边。它们增加计算预算，不保证每项指标单调变好。CUDA 档位是另一套配置：7,000 / 15,000 / 22,000 步，不能与本表的 MPS 时间混用。", "",
          "优化耗时不包含数据下载、COLMAP、神经网络分割等完整流水线。快速模式可能受首次加载开销影响，不能由这里的秒数推断大型场景需要的分钟数。", "",
          "## 重要物品精细重建", "",
          "选取 `stuffed bear`，使用训练视角预测掩码加强。对照双方均为 80 步、160 个高斯、96 像素训练；区域 RGB 损失权重由 1 提升到 4，同时提高区域内初始点的采样概率。测试 ROI 来自 5 个留出视角的人工标注。", "",
          "| 指标 | 均衡基线 | 加强熊玩偶 | 变化 |", "|---|---:|---:|---:|"]
for section,key,title in [("priority_roi","mean_psnr_db","目标区域 PSNR / dB"),("priority_roi","mean_ssim","目标区域 SSIM"),("rgb","mean_psnr_db","整张图像 PSNR / dB"),("rgb","mean_ssim","整张图像 SSIM")]:
    a,b=base[section][key],priority[section][key]
    lines.append(f"| {title} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
lines += ["", "因此，重点优化已经进入真实训练，并在这个场景上改善目标区域；有限预算从其他区域转移，使全图质量下降。这不是新增几何细分或完整自适应 densification 的证明。", "",
          "## 语义与边界", "",
          f"GroundingDINO Tiny + SAM ViT Base 在 8 张训练图片上生成 {len(tea['semantics_training']['masks'])} 个预测掩码，实际耗时 {tea['semantics_training']['seconds']:.2f} 秒；测试图片未送入这两个模型。融合后的高斯在留出视角渲染类别掩码，与 6 张图像、14 类、59 个类别与视角组合的人工 GT 比较。", "",
          "下表是**留出视角二维投影掩码 IoU**，不是 3D 点云 IoU。全部类别都计入，漏检记 0。边界采用 128 像素长边下向内 2 像素宽的边界带。", "",
          "| 模式 | 59 组合平均 IoU | 14 类等权 mIoU | 边界 IoU |", "|---|---:|---:|---:|"]
for r in tea["runs"]:
    m=r["semantics"]
    lines.append(f"| {r['name']} | {100*m['mean_iou']:.3f}% | {100*m['class_mean_iou']:.3f}% | {100*m['mean_boundary_iou']:.3f}% |")
groups=base["semantics"]["image_border_groups"]
lines += ["", "均衡模式的触边与非触边分组：", "", "| GT 分组 | 类别与视角组合 | 投影 IoU | 边界 IoU |", "|---|---:|---:|---:|"]
for k,title in [("touching_image_border","触及画面边界"),("inside_image","未触及画面边界")]:
    g=groups[k]
    lines.append(f"| {title} | {g['object_view_pairs']} | {g['mean_iou']*100:.3f}% | {g['mean_boundary_iou']*100:.3f}% |")
lines += ["", "当前语义精度很低。修复后精细模式虽有高斯获得标签，但所有测试类别掩码在固定 alpha 阈值 0.5 下均为空；不能因训练步数增加而声称更准确。小规模高斯表示、遮挡估计、预测掩码和融合置信规则都需要继续改进。", "",
          "边界修复已通过代码回归检查，但本数据未证明它提升总体 IoU。代码选择对不确定归属保留候选，避免直接把相邻物体错误合并或删除；这项行为修复与定量精度提升是两个不同结论。", "",
          "## 复核材料", "",
          "- `results/bonsai/summary.json`：图像划分、所有视角 RGB 结果。", "- `results/teatime/summary.json`：当前核心代码哈希、语义/边界分组、区域加强对照。", "- `results/teatime-before-boundary-fix/summary.json`：旧融合器的独立历史结果。", "- 各模式目录下的 `scene.json`、`metrics.json` 和 `renders/`：可复核场景与失败案例。", "- `README.zh-CN.md`：指标定义、泄露限制和运行命令。", "",
          "目前只完成单种子、两场景、低分辨率的定量评价；没有毫米级几何精度、LPIPS、部件层级正确率或实例跨视角一致性成绩。CUDA 评估脚本已准备，须以实际 GPU 执行输出为准。"]
(ROOT/"RESULTS.zh-CN.md").write_text("\n".join(lines)+"\n")

# A contact sheet uses the saved low-resolution diagnostic strips. Images are
# enlarged for readability, with no enhancement or generated detail.
try:
    font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",19)
except OSError:
    font=ImageFont.load_default()
thumb_w,thumb_h,left,top=256,192,160,44
sheet=Image.new("RGB",(left+thumb_w*5,top+thumb_h*4),(246,247,249))
draw=ImageDraw.Draw(sheet)
columns=["Held-out GT","Fast","Balanced","Fine","Priority"]
for j,title in enumerate(columns):
    draw.text((left+j*thumb_w+8,10),title,font=font,fill=(20,30,40))
row=0
for scene in ["bonsai","teatime"]:
    for frame in [0,3]:
        draw.text((10,top+row*thumb_h+18),f"{scene}\nview {frame+1}",font=font,fill=(20,30,40))
        for col,mode in enumerate(["fast","fast","balanced","fine","balanced-priority"]):
            files=sorted((RESULTS/scene/mode/"renders").glob(f"{frame:02d}-*.png"))
            files=[p for p in files if not p.name.endswith("priority-mask.png")]
            if not files:
                draw.text((left+col*thumb_w+8,top+row*thumb_h+70),"Not tested",font=font,fill=(100,100,100))
                continue
            with Image.open(files[0]) as strip:
                width=strip.width//3
                column=0 if col==0 else 1
                image=strip.crop((column*width,24,(column+1)*width,strip.height))
                image.thumbnail((thumb_w,thumb_h),Image.Resampling.NEAREST)
                # Preserve pixel-level appearance when enlarging exactly 2x.
                image=image.resize((width*2,image.height*2),Image.Resampling.NEAREST)
                sheet.paste(image,(left+col*thumb_w,top+row*thumb_h))
        row+=1
sheet.save(RESULTS/"quality-contact-sheet.png")
print(ROOT/"RESULTS.zh-CN.md")
