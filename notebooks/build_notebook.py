import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

# ── Header ────────────────────────────────────────────────────────────────────
md([
    "# ScienceQA Ablation Study — Analysis\n",
    "**ECE 9660 · Western University**\n\n",
    "Models: `llama3.2:3b` · `phi4-mini` · `llama3.1:8b` · `mixtral:8x7b` · `claude-haiku`  \n",
    "Conditions: C0 (No Context) · C1 (+Lecture) · C2 (+Hint) · C3 (+Grade) · C4 (Full)  \n",
    "Sample: 375 examples · grades 3-7 · 75/grade · text-only · all have lecture+hint+solution"
])

# ── Setup ─────────────────────────────────────────────────────────────────────
code([
    "import json, glob, sys, os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "plt.rcParams.update({'figure.dpi':130,'font.family':'DejaVu Sans',\n",
    "                     'axes.spines.top':False,'axes.spines.right':False})\n",
    "PURPLE='#4F2683'; BLUE='#2980B9'; ORANGE='#E67E22'; GREEN='#27AE60'; RED='#C0392B'\n",
    "MODEL_ORDER  = ['llama3.2-3b','phi4-mini','llama3.1-8b','mixtral-8x7b','claude-haiku']\n",
    "MODEL_LABELS = ['Llama 3.2 3B','Phi-4 Mini','Llama 3.1 8B','Mixtral 8x7B','Claude Haiku']\n",
    "MODEL_COLORS = dict(zip(MODEL_ORDER,[BLUE,ORANGE,PURPLE,GREEN,RED]))\n",
    "COND_ORDER   = ['C0','C1','C2','C3','C4']\n",
    "COND_LABELS  = {'C0':'C0\\nNo Ctx','C1':'C1\\n+Lecture','C2':'C2\\n+Hint',\n",
    "                'C3':'C3\\n+Grade','C4':'C4\\nFull'}\n",
    "GRADE_ORDER  = ['grade3','grade4','grade5','grade6','grade7']\n",
    "GRADE_LABELS = ['Grade 3','Grade 4','Grade 5','Grade 6','Grade 7']\n",
    "RESULTS_DIR  = '../results'\n",
    "print('Setup complete.')"
])

# ── 1. Load results ───────────────────────────────────────────────────────────
md(["## 1 · Load Results"])
code([
    "rows = []\n",
    "seen = {}\n",
    "for f in sorted(glob.glob(f'{RESULTS_DIR}/*.json')):\n",
    "    with open(f, encoding='utf-8') as fp:\n",
    "        d = json.load(fp)\n",
    "    model = d.get('model', '')\n",
    "    cond  = d.get('condition', '')\n",
    "    if not model or not cond:\n",
    "        continue\n",
    "    seen[(model, cond)] = d\n",
    "for (model, cond), d in seen.items():\n",
    "    for r in d['results']:\n",
    "        rows.append(r)\n",
    "df = pd.DataFrame(rows)\n",
    "df['model'] = df['model'].str.replace(\n",
    "    'claude-haiku-4-5-20251001', 'claude-haiku', regex=False)\n",
    "df = df[df['model'].isin(MODEL_ORDER) & df['grade'].isin(GRADE_ORDER)]\n",
    "print(f'Rows: {len(df)}')\n",
    "print(f'Models: {sorted(df.model.unique())}')\n",
    "print(f'Conditions: {sorted(df.condition.unique())}')"
])

# ── 2. Accuracy table ─────────────────────────────────────────────────────────
md(["## 2 · Accuracy Table (Model x Condition)"])
code([
    "acc = (\n",
    "    df.groupby(['model','condition'])['is_correct']\n",
    "    .mean().unstack('condition')\n",
    "    .reindex(index=MODEL_ORDER, columns=COND_ORDER)\n",
    "    .mul(100).round(1)\n",
    ")\n",
    "acc.index = MODEL_LABELS\n",
    "acc.columns.name = None\n",
    "for c in ['C1','C2','C3','C4']:\n",
    "    acc[f'D{c}'] = (acc[c] - acc['C0']).round(1)\n",
    "print('Accuracy (%) and delta vs C0:')\n",
    "display(acc)"
])

# ── 3. Figure 1 — heatmap ─────────────────────────────────────────────────────
md(["## 3 · Figure 1 — Accuracy Heatmap (Model x Condition)"])
code([
    "fig, ax = plt.subplots(figsize=(7,4))\n",
    "sns.heatmap(acc[COND_ORDER], annot=True, fmt='.1f', cmap='YlOrRd',\n",
    "            linewidths=0.5, ax=ax, vmin=60, vmax=100,\n",
    "            cbar_kws={'label':'Accuracy (%)'})\n",
    "ax.set_title('Answer Accuracy (%) — Model x Condition', fontweight='bold', pad=10)\n",
    "ax.tick_params(axis='y', rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{RESULTS_DIR}/fig1_accuracy_heatmap.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved fig1_accuracy_heatmap.png')"
])

# ── 4. Figure 2 — line plot ───────────────────────────────────────────────────
md(["## 4 · Figure 2 — Accuracy by Condition (Line Plot)"])
code([
    "fig, ax = plt.subplots(figsize=(9,5))\n",
    "x = np.arange(len(COND_ORDER))\n",
    "for model, label in zip(MODEL_ORDER, MODEL_LABELS):\n",
    "    vals = [acc.loc[label, c] for c in COND_ORDER]\n",
    "    ax.plot(x, vals, marker='o', linewidth=2, markersize=7,\n",
    "            color=MODEL_COLORS[model], label=label)\n",
    "    for xi, v in zip(x, vals):\n",
    "        ax.text(xi, v+0.8, f'{v:.1f}', ha='center', va='bottom',\n",
    "                fontsize=7.5, color=MODEL_COLORS[model])\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])\n",
    "ax.set_ylabel('Accuracy (%)')\n",
    "ax.set_title('Accuracy by Condition per Model', fontweight='bold', pad=10)\n",
    "ax.set_ylim(55, 108)\n",
    "ax.grid(axis='y', linestyle='--', alpha=0.4)\n",
    "ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{RESULTS_DIR}/fig2_accuracy_by_condition.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved fig2_accuracy_by_condition.png')"
])

# ── 5. Figure 3 — delta bars ──────────────────────────────────────────────────
md(["## 5 · Figure 3 — Condition Effect (Delta vs C0 Baseline)"])
code([
    "x = np.arange(4)\n",
    "w = 0.15\n",
    "offsets = np.linspace(-2, 2, len(MODEL_ORDER)) * w\n",
    "fig, ax = plt.subplots(figsize=(9,4.5))\n",
    "for i, (model, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):\n",
    "    vals = [acc.loc[label, f'D{c}'] for c in ['C1','C2','C3','C4']]\n",
    "    bars = ax.bar(x+offsets[i], vals, width=w,\n",
    "                  color=MODEL_COLORS[model], label=label, alpha=0.88, zorder=3)\n",
    "    for bar, v in zip(bars, vals):\n",
    "        ax.text(bar.get_x()+bar.get_width()/2,\n",
    "                bar.get_height()+(0.3 if v>=0 else -1.2),\n",
    "                f'{v:+.1f}', ha='center', va='bottom', fontsize=7)\n",
    "ax.axhline(0, color='black', linewidth=0.8)\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(['+Lecture (C1)','+Hint (C2)','+Grade (C3)','Full (C4)'])\n",
    "ax.set_ylabel('Accuracy Change (pp vs C0)')\n",
    "ax.set_title('Condition Effect Relative to C0 Baseline', fontweight='bold', pad=10)\n",
    "ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)\n",
    "ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=8)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{RESULTS_DIR}/fig3_condition_delta.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved fig3_condition_delta.png')"
])

# ── 6. Figure 4 — per-grade bars ─────────────────────────────────────────────
md(["## 6 · Figure 4 — Per-Grade Accuracy (C0 vs C2 vs C4)"])
code([
    "compare = ['C0','C2','C4']\n",
    "col = {'C0':BLUE, 'C2':ORANGE, 'C4':PURPLE}\n",
    "fig, axes = plt.subplots(1, 5, figsize=(18,4), sharey=True)\n",
    "for ax, model, label in zip(axes, MODEL_ORDER, MODEL_LABELS):\n",
    "    sub = df[df.model == model]\n",
    "    x = np.arange(5); w = 0.25; off = [-1, 0, 1]\n",
    "    for o, cond in zip(off, compare):\n",
    "        s2 = sub[sub.condition == cond]\n",
    "        vals = [s2[s2.grade==g]['is_correct'].mean()*100 for g in GRADE_ORDER]\n",
    "        ax.bar(x+o*w, vals, width=w, color=col[cond], label=cond, alpha=0.85, zorder=3)\n",
    "    ax.set_title(label, fontweight='bold', fontsize=9)\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(['G3','G4','G5','G6','G7'], fontsize=8)\n",
    "    ax.set_ylim(40, 112)\n",
    "    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)\n",
    "    if ax is axes[0]:\n",
    "        ax.set_ylabel('Accuracy (%)')\n",
    "        ax.legend(fontsize=8)\n",
    "fig.suptitle('Per-Grade Accuracy: C0 vs C2 (+Hint) vs C4 (Full)',\n",
    "             fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{RESULTS_DIR}/fig4_accuracy_by_grade.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved fig4_accuracy_by_grade.png')"
])

# ── 7. Figure 5 — grade x condition heatmaps ─────────────────────────────────
md(["## 7 · Figure 5 — Grade x Condition Heatmap per Model"])
code([
    "fig, axes = plt.subplots(1, 5, figsize=(20,3.5))\n",
    "for ax, model, label in zip(axes, MODEL_ORDER, MODEL_LABELS):\n",
    "    sub = df[df.model == model]\n",
    "    heat = (\n",
    "        sub.groupby(['grade','condition'])['is_correct']\n",
    "        .mean().unstack('condition')\n",
    "        .reindex(index=GRADE_ORDER, columns=COND_ORDER)\n",
    "        .mul(100).round(1)\n",
    "    )\n",
    "    heat.index = GRADE_LABELS\n",
    "    sns.heatmap(heat, annot=True, fmt='.0f', cmap='YlOrRd',\n",
    "                vmin=50, vmax=100, linewidths=0.4,\n",
    "                cbar=(ax is axes[-1]), ax=ax, annot_kws={'size':8})\n",
    "    ax.set_title(label, fontweight='bold', fontsize=9)\n",
    "    ax.set_xlabel('Condition')\n",
    "    ax.set_ylabel('')\n",
    "    ax.tick_params(axis='y', rotation=0)\n",
    "fig.suptitle('Accuracy (%) by Grade x Condition', fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{RESULTS_DIR}/fig5_grade_condition_heatmap.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved fig5_grade_condition_heatmap.png')"
])

# ── 8. Consistency ────────────────────────────────────────────────────────────
md(["## 8 · Choice Consistency"])
code([
    "cons = (\n",
    "    df.assign(p=df['predicted_answer'].notna())\n",
    "    .groupby(['model','condition'])['p'].mean()\n",
    "    .unstack('condition')\n",
    "    .reindex(index=MODEL_ORDER, columns=COND_ORDER)\n",
    "    .mul(100).round(1)\n",
    ")\n",
    "cons.index = MODEL_LABELS\n",
    "print('Choice Consistency (%) -- fraction of responses with parseable answer letter:')\n",
    "display(cons)"
])

# ── 9. RQ analysis ────────────────────────────────────────────────────────────
md(["## 9 · RQ Analysis"])
code([
    "for rq, cond, title in [\n",
    "    ('RQ0', 'C0', 'Parametric baseline (C0 -- no context)'),\n",
    "    ('RQ1', 'C1', 'Lecture grounding effect (C1 - C0)'),\n",
    "    ('RQ2', 'C2', 'Hint effect (C2 - C0)'),\n",
    "    ('RQ3', 'C3', 'Grade context effect (C3 - C0)'),\n",
    "]:\n",
    "    print(f'=== {rq}: {title} ===')\n",
    "    for label in MODEL_LABELS:\n",
    "        if cond == 'C0':\n",
    "            print(f'  {label:<20}: {acc.loc[label, cond]:.1f}%')\n",
    "        else:\n",
    "            d = acc.loc[label, f'D{cond}']\n",
    "            verdict = 'HELPS' if d > 0 else 'HURTS'\n",
    "            print(f'  {label:<20}: {d:+.1f} pp  ({verdict})')\n",
    "    print()"
])

# ── 10. Qualitative ───────────────────────────────────────────────────────────
md(["## 10 · Qualitative Examples"])
code([
    "def show(row, lbl=''):\n",
    "    labs = list('ABCDE')[:len(row['choices'])]\n",
    "    gold = labs[int(row['gold_answer'])]\n",
    "    pred = labs[int(row['predicted_answer'])] if (row['predicted_answer'] is not None and row['predicted_answer'] == row['predicted_answer']) else '?'\n",
    "    tag = 'CORRECT' if row['is_correct'] else 'WRONG'\n",
    "    print(f'--- {lbl} [{tag}] | {row[\"grade\"]} ---')\n",
    "    print(f'Q: {row[\"question\"]}')\n",
    "    for l, c in zip(labs, row['choices']):\n",
    "        print(f'  {l}. {c}{\" <- ans\" if l == gold else \"\"}')\n",
    "    print(f'Pred: {pred}')\n",
    "    print(str(row['response'])[:300])\n",
    "    print()\n",
    "\n",
    "best = 'claude-haiku'\n",
    "print('=== CORRECT -- claude-haiku C4 ===')\n",
    "for _, r in df[(df.model==best)&(df.condition=='C4')&df.is_correct].sample(2, random_state=1).iterrows():\n",
    "    show(r, 'C4')\n",
    "print('=== FAILURES -- claude-haiku C0 ===')\n",
    "for _, r in df[(df.model==best)&(df.condition=='C0')&~df.is_correct].sample(2, random_state=1).iterrows():\n",
    "    show(r, 'C0')"
])

code([
    "# Grade conditioning: same Grade-3 question, C0 vs C4\n",
    "sub = df[df.model == 'claude-haiku']\n",
    "q0 = set(sub[(sub.condition=='C0')&(sub.grade=='grade3')]['question'])\n",
    "q4 = set(sub[(sub.condition=='C4')&(sub.grade=='grade3')]['question'])\n",
    "common = list(q0 & q4)\n",
    "if common:\n",
    "    q = common[0]\n",
    "    r0 = sub[(sub.condition=='C0')&(sub.question==q)].iloc[0]\n",
    "    r4 = sub[(sub.condition=='C4')&(sub.question==q)].iloc[0]\n",
    "    print(f'Q: {q}\\n')\n",
    "    print('--- C0 (no context) ---')\n",
    "    print(str(r0['response'])[:400])\n",
    "    print('\\n--- C4 (full context + grade 3) ---')\n",
    "    print(str(r4['response'])[:400])"
])

# ── 11. Summary ───────────────────────────────────────────────────────────────
md(["## 11 · Summary of Key Findings"])
code([
    "print('='*65)\n",
    "print('KEY FINDINGS')\n",
    "print('='*65)\n",
    "print('\\n[C4 Full Context accuracy by model]')\n",
    "for label in MODEL_LABELS:\n",
    "    print(f'  {label:<20}: {acc.loc[label, \"C4\"]:.1f}%')\n",
    "print('\\n[Most impactful condition per model (delta vs C0)]')\n",
    "for label in MODEL_LABELS:\n",
    "    ds = {c: acc.loc[label, f'D{c}'] for c in ['C1','C2','C3','C4']}\n",
    "    bc = max(ds, key=ds.get)\n",
    "    print(f'  {label:<20}: {bc} ({ds[bc]:+.1f} pp)')\n",
    "print('\\n[Lecture effect: does lecture help or hurt accuracy?]')\n",
    "for label in MODEL_LABELS:\n",
    "    d = acc.loc[label, 'DC1']\n",
    "    print(f'  {label:<20}: {d:+.1f} pp  ({\"HELPS\" if d > 0 else \"HURTS\"})')\n",
    "print('\\n[Grade 6 drop vs Grade 5 -- C4 full context]')\n",
    "for model, label in zip(MODEL_ORDER, MODEL_LABELS):\n",
    "    s = df[(df.model==model) & (df.condition=='C4')]\n",
    "    g5 = s[s.grade=='grade5']['is_correct'].mean()*100\n",
    "    g6 = s[s.grade=='grade6']['is_correct'].mean()*100\n",
    "    print(f'  {label:<20}: G5={g5:.1f}% G6={g6:.1f}% drop={g6-g5:+.1f} pp')\n",
    "print('='*65)"
])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook written successfully.")
