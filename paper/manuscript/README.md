# Manuscript

`main.tex` 是唯一可编辑正文，`main.pdf` 是其当前构建结果。正文仍以旧 canonical 稿为起点，尚未完成 RSM 新方法与最新实验数字的整合；具体修改顺序见 `REVISION_PLAN.md`。

## 文件职责

- `REVISION_PLAN.md`：当前写作计划和已经核实的方法边界。
- `reference/current_review_copy_Reskip.pdf`：只读审阅参考，不是正文源。
- `figures/`：仅保留当前旧稿仍实际引用的四张历史图。
- `../figures/main/`、`../figures/appendix/`：RSM 新稿的 canonical 图源。
- `references.bib`、`neurips_2026.sty`：独立编译依赖。

正文数字统一从 `../PAPER_NUMBERS.md` 取用。旧投稿、旧中文稿和 pre-revision render 已在其他 canonical/归档位置保存，不在活动目录重复存放。

运行 `./compile.sh` 可在服务器上构建正文；成功后只保留 `main.pdf`，LaTeX 中间文件由脚本清理。
