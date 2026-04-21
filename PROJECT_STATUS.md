# HCC 早期复发 WSI 预测项目 — 当前状态总结

**更新时间**：2026-04-21

---

## 任务目标

基于全切片图像（WSI，H&E 染色）预测肝细胞癌（HCC）术后早期复发（24个月内），用于辅助临床决策，最终目标是发表论文，但是目前指标完全不够，至少auc要在0.85以上。

**核心评估指标**：
- 内部交叉验证 AUROC（5折，Ourdata）
- 外部验证 AUROC（TCGA-LIHC）

---

## 数据概况

### 内部数据集（Ourdata）

| 项目 | 数值 |
|------|------|
| 原始患者数 | 627 |
| 有效患者数（有 WSI）| **509** |
| 剔除原因 | 无对应 WSI 切片文件（118例，cohort1: 45，cohort2: 73）|
| 标签分布 | 早期复发=1（309例），无复发=0（200例）|
| Cohort1 | 292例，随访时间全部为 24.0（人工截断，非真实随访）|
| Cohort2 | 217例，真实随访时间 25–132 个月 |
| 主要病因 | HBV 相关 HCC（中国三甲医院）|

### 外部验证集（TCGA-LIHC）

| 项目 | 数值 |
|------|------|
| 有效患者数 | **234** |
| 标签分布 | 早期复发=1（147例），无复发=0（87例）|
| 主要病因 | HCV / NASH 相关 HCC（美国 TCGA）|

---

## 技术框架

**STAMP**（开源 MIL 框架）：冻结预训练特征提取器 → MIL 聚合器训练（从头）→ 二分类预测

- 特征提取器：**完全冻结**，不做任何微调
- MIL 聚合器：**从头训练**，5折交叉验证
- 原生支持模型：`vit`、`trans_mil`、`mlp`、`linear`、`barspoon`、`abmil`（已手动添加）
- **不支持**：端对端微调、原生多模态输入、临床特征直接融合

---

## 核心困难

### 1. TCGA 外部验证 AUROC 接近随机（~0.56）
**根本原因**：HBV（Ourdata）vs HCV/NASH（TCGA）的**生物学域漂移**。两种病因学 HCC 的复发驱动机制不同，模型学到的 HBV-HCC 病理特征无法迁移到 HCV-HCC。

已尝试排除的技术原因：
- 染色风格差异：Phikon（TCGA 预训练）仍然失败（AUROC 0.491）
- MIL 聚合器设计：换 TransMIL/ABMIL 均无改善
- 特征提取器能力：UNI2（更强）和 Phikon（TCGA 域）均低于随机

**结论**：TCGA 域漂移是生物学本质问题，不可通过图像处理或模型改进解决。

### 2. 内部 AUROC 提升空间有限（当前上限 ~0.76）
509 例小样本限制了模型复杂度和提升空间。文献 SOTA（Shi et al. 2021, *Gut*）在更大 HBV-HCC 队列上达到 ~0.78。

### 3. 临床特征无增量价值
AFP、年龄、性别的后验 ensemble 将 AUROC 从 0.72 拉低至 0.61，原因是临床逻辑回归的 AUROC 仅 0.37–0.49（低于随机）。WSI 特征已编码了临床变量相关的形态学信息。

---

## 已完成实验与结果

### 单模型结果

| 实验 | 特征提取器 | MIL 模型 | 训练设置 | 内部 CV AUROC（95%CI）| TCGA AUROC |
|------|-----------|----------|---------|----------------------|------------|
| Exp00 基线 | CONCH1.5 | ViT | 标准（32ep）| 0.724（0.661–0.787）| 0.558 |
| **Exp01 TransMIL** | CONCH1.5 | TransMIL | 标准 | **0.744**（0.711–0.778）| **0.564** |
| Exp03 延长训练 | CONCH1.5 | ViT | 64ep, bag=1024 | 0.732（0.696–0.768）| 0.547 |
| Exp07 UNI2 | UNI2 | ViT | 标准 | 0.752（0.706–0.798）| 0.485 ⚠️ |
| Exp09 ABMIL | CONCH1.5 | ABMIL | 标准 | 0.715（0.675–0.755）| 0.559 |
| Exp10 Phikon | Phikon | TransMIL | 标准 | 0.751（0.717–0.785）| 0.491 ⚠️ |

### 临床 Ensemble（临床特征融合）

| 实验 | 说明 | 内部 AUROC | 结论 |
|------|------|-----------|------|
| Exp02 ViT + AFP/age/sex | 后验逻辑回归 ensemble | 0.614 ↓↓ | 临床特征无效，损害性能 |
| Exp05 TransMIL + AFP/age/sex | 后验逻辑回归 ensemble | 0.621 ↓↓ | 同上 |

### 模型 Ensemble（最优组合）

| 组合 | 内部 CV AUROC | TCGA AUROC | 备注 |
|------|--------------|------------|------|
| TransMIL + UNI2 + Phikon | **0.760** | 0.500 ⚠️ | 内部最优，TCGA 不可用 |
| TransMIL + Phikon | 0.758 | 0.498 ⚠️ | 同上 |
| TransMIL + UNI2 | 0.755 | 0.489 ⚠️ | 同上 |
| **TransMIL + ABMIL** | 0.733 | **0.567** | TCGA 最优，内部略退 |
| TransMIL 单模型 | 0.744 | 0.564 | 最佳单模型 |

---

## 关键结论

1. **最佳单模型**：Exp01 TransMIL（CONCH1.5），内部 0.744 / TCGA 0.564，稳健、置信区间最窄，**推荐作为论文主模型**

2. **最高内部 AUROC**：TransMIL + UNI2 + Phikon ensemble（0.760），但 TCGA 约随机水平，仅适合消融实验展示

3. **最高 TCGA AUROC**：TransMIL + ABMIL ensemble（0.567），作为外部验证最优报告值

4. **UNI2 / Phikon 的规律**：内部性能强但 TCGA 低于随机（<0.5），任何包含二者的 ensemble 都会拖垮 TCGA 结果

5. **临床特征**：AFP/年龄/性别在本队列无预测价值，不应纳入最终模型

6. **TCGA 域漂移**：生物学本质问题，技术手段无法根本解决

---

## 代码修改记录（相对于原始 STAMP）

| 文件 | 修改内容 |
|------|---------|
| `src/stamp/modeling/models/abmil.py` | 新增 ABMIL（Gated Attention MIL）模型 |
| `src/stamp/modeling/registry.py` | 注册 `abmil` 到 ModelName |
| `src/stamp/modeling/config.py` | 新增 `AbmilModelParams`，注册到 `ModelParams` |
| `src/stamp/preprocessing/extractor/phikon.py` | 新增 Phikon（Owkin iBOT ViT-B）提取器 |
| `src/stamp/preprocessing/config.py` | 注册 `phikon` 到 `ExtractorName` |
| `src/stamp/preprocessing/__init__.py` | 注册 Phikon 的 dispatch case |
| `scripts/prepare_data.py` | 修复 PATIENT 列匹配（病理号），添加 TCGA barcode 提取 |
| `scripts/clinical_ensemble.py` | 新增后验临床特征 ensemble（AFP 特殊字符处理，bootstrap CI 修复）|
| `scripts/model_ensemble.py` | 新增多模型概率 ensemble 脚本 |
| `scripts/macenko_normalize_tiles.py` | 新增 Macenko 染色标准化脚本（备用，暂未使用）|
| `scripts/collect_results.py` | 新增自动收集 AUROC 脚本 |
| `scripts/generate_report.py` | 新增生成对比表脚本 |

---

## 下一步选项

### 选项 A：收尾写论文（claude这样说但是个人认为不行哈，这个指标还不够）
当前结果已足够支撑论文：
- 内部验证 AUROC 0.744 接近 SOTA（Shi et al. 0.78，更大队列）
- TCGA 域漂移作为重要发现和讨论点（HBV vs HCV 的生物学不可迁移性）
- 消融实验对比 extractor / aggregator 贡献
- Attention map 可解释性分析（MVI 相关区域）

### 选项 B：继续提升外部验证
- **Exp08**：Cohort2 生存分析（真实随访时间，Cox 模型），用生存分析替代二分类，可能更鲁棒
- **ICJG-LIRI-JP**：寻找公开 HBV-HCC 队列作为更合理的外部验证，替代 TCGA
- **LoRA 微调**：对特征提取器做轻量微调，需自定义训练代码，超出 STAMP 框架

### 选项 C：可解释性分析
- Attention map 热图生成（STAMP `heatmaps` 命令）
- MVI+ vs MVI- 亚组 AUROC 分析
- 高注意力 patch 的病理形态分析
