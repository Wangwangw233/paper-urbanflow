# 补充材料图目录（figures_supp/）

`supplement.tex` 的 `\graphicspath` 为 `{figures/}{figures_supp/}`。补充材料只引用下列 7 个规范 PDF；正文图仍位于 `figures/`。图号由 LaTeX 计数器自动生成，标签与文件名按下表一一对应。

## 规范文件、标签与位置

| 补充图 | 规范文件 | LaTeX 标签 | 首次出现位置 | 结果解释位置 | 内容与冻结数据来源 |
|---|---|---|---|---|---|
| Fig. S1 | `figS01_joint_spectrum_2d.pdf` | `fig:supp-1` | S3 `note:spectral` | S7 `note:spectral-results` | 三城源/路由二维联合谱；`reports/threecity_joint_spectrum/manuscript_revision/verified_joint.npz` 与 `redraw_joint_spectrum.py` |
| Fig. S2 | `figS02_heldout_daily_scores.pdf` | `fig:supp-2` | S3 `note:spectral` | S6/S7 `note:coupling-results` / `note:spectral-results` | 三城逐日 P/D；`reports/mainstream_mobility_comparison/results/s1/*.json`，保留 San Francisco 的五折结构 |
| Fig. S3 | `figS03_coherent_response_train_test.pdf` | `fig:supp-3` | S3 `note:spectral` | S7 `note:spectral-results` | 三城 TRAIN/TEST coherent response；`obj_*.json` 与 `reports/threecity_redesign/results/chengdu_response18.npz`，冻结 P-selected operator，Chengdu 使用全部 10 个 18 点 DFT bin；成都当前文件仅含 TEST，TRAIN 需同协议导出 |
| Fig. S4 | `figS04_anisotropy_distributions.pdf` | `fig:supp-4` | S4 `note:spatial` | S8 `note:anisotropy-results` | 三城 response/road-support 各向异性 CDF；冻结各向异性数组与 `reports/threecity_figure5_revision/results/chengdu_anisotropy_summary.json` |
| Fig. S5 | `figS05_matched_capacity.pdf` | `fig:supp-5` | S4 `note:spatial` | S9 `note:structural-results` | 三城 matched-capacity 下 road、mobility-inferred 与 permuted mobility medium；`reports/threecity_figure5_revision/results/partF_summary.json` |
| Fig. S6 | `figS06_heldout_reconstruction.pdf` | `fig:supp-6` | S6 `note:coupling-results` | S6 `note:coupling-results` | 三城 P 帧 held-out physical reconstruction，使用 raw TEST inputs/targets；Porto/SF 保留冻结 picked indices/phases；成都 `chengdu_snapshot18.npz` 使用 `picked_i=2`、`picked_phase=6`（12:00）、`picked_date=16304`，并通过 pinned-basis/`R_P` 校验 |
| Fig. S7 | `figS07_nonlinear_exploratory.pdf` | `fig:supp-7` | S7 `note:spectral-results` | S7 `note:spectral-results` | 成都 18 相位全 FACE 非线性探索；`reports/chengdu_nonlinear_exploratory/results/chengdu_nonlinear_exploratory.json` 与 `validation.json`；仅 3 TEST 日期、144 列/144 TRAIN 行饱和，明确标记 underpowered |

## 引用与排版约定

- 七图按首次出现顺序排列：S3 中依次放置 Fig. S1–S3，S4 中依次放置 Fig. S4–S5，S6 中放置 Fig. S6，S7 中放置 Fig. S7。这样无需手动播种计数器，即可保持图号、标签和规范文件名一致。
- 引用一律使用 `\ref{fig:supp-1}` 至 `\ref{fig:supp-7}`，不要在正文中手写图号。结果 Note 可以引用此前方法 Note 中首次出现的图。
- 每个 Note 由 `\suppnote` 自动计数并在边界执行 `\FloatBarrier`；图件放在相关定义或结果附近，不设置独立图件堆叠区。
- 图文件是编译输入。重建图件与重算实验是两个步骤；不得把冻结图件的存在表述为原始轨迹到结果的端到端复现。
- Fig. S6 只用于三城 P 帧定性展示，完整定量结论仍以 `tab:crosscity` 的全部 held-out 观测和 pooled-error 口径为准。
- Fig. S7 只提供探索性证据。三日 TEST 的 bootstrap 区间不能克服饱和设计和有限独立日期信息，不得据此声称成都存在或不存在非线性。
