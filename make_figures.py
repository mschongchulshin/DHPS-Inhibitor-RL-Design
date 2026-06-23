"""
Publication-quality figures for DHPS MMGBSA binding prediction paper
Nature Communications style
"""
import warnings; warnings.filterwarnings("ignore")
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── rcParams ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# ── Color palette ────────────────────────────────────────────────────────────
C_BEST   = '#0D47A1'   # darkest blue — selected model (Ridge, BO)
C_BLUE1  = '#1565C0'   # dark blue
C_BLUE2  = '#1976D2'   # medium blue
C_TEAL1  = '#00897B'   # dark teal
C_TEAL2  = '#26A69A'   # medium teal
C_TEAL3  = '#80CBC4'   # light teal
C_GRAY1  = '#607D8B'   # dark gray
C_GRAY2  = '#90A4AE'   # medium gray
C_GRAY3  = '#CFD8DC'   # light gray
C_LINE   = '#37474F'   # axes/reference line

MODEL_COLORS = {
    "GNN":       C_BLUE2,
    "Ridge":     C_BEST,    # selected
    "LightGBM":  C_TEAL1,
    "XGBoost":   C_TEAL2,
    "BiLSTM":    C_TEAL3,
    "ChemBERTa": C_GRAY2,
    "SVR":       C_GRAY3,
}

METHOD_COLORS = {
    "GA":        C_GRAY2,
    "BO":        C_BEST,    # selected
    "RNN":       C_TEAL2,
    "Fragment":  C_TEAL3,
    "Retrieval": C_GRAY3,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

def nc_style(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.tick_params(which='both', direction='in',
                   top=True, right=True, length=3.5, width=0.8)

def save_fig(fig, path_no_ext):
    fig.savefig(path_no_ext + '.png')
    fig.savefig(path_no_ext + '.svg')
    print(f"Saved: {path_no_ext}.png / .svg")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Forward Model 31-seed Violin (Ridge + ML models)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    with open("results/ridge_31seed_result.json") as f:
        ridge_d = json.load(f)
    with open("results/ml_31seed_result.json") as f:
        ml_d = json.load(f)
    with open("results/gnn_31seed_result.json") as f:
        gnn_d = json.load(f)
    with open("results/bilstm_31seed_result.json") as f:
        bl_d = json.load(f)

    # All 6 models with 31-seed data, consistent ordering
    model_data = {
        "Ridge":    ridge_d["seeds"],
        "GNN":      gnn_d["seeds"],
        "BiLSTM":   bl_d["seeds"],
        "LightGBM": ml_d["LightGBM"]["seeds"],
        "XGBoost":  ml_d["XGBoost"]["seeds"],
        "SVR":      ml_d["SVR"]["seeds"],
    }

    models   = list(model_data.keys())
    colors_v = [MODEL_COLORS.get(m, C_GRAY2) for m in models]

    # R² → supplementary; main figure: RMSE + Spearman only
    metrics = [
        ("RMSE",     "RMSE (kcal/mol)", "a"),
        ("Spearman", "Spearman ρ",      "b"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    fig.subplots_adjust(wspace=0.42)

    for ax, (key, ylabel, lbl) in zip(axes, metrics):
        all_vals = [[r[key] for r in model_data[m]] for m in models]
        positions = list(range(len(models)))

        vp = ax.violinplot(all_vals, positions=positions, widths=0.45,
                           showmeans=False, showmedians=False, showextrema=False)
        for body, c in zip(vp["bodies"], colors_v):
            body.set_facecolor(c); body.set_alpha(0.35)
            body.set_edgecolor(c); body.set_linewidth(0.8)

        for i, (vals, c) in enumerate(zip(all_vals, colors_v)):
            ax.scatter([i]*len(vals), vals, color=c, s=6, alpha=0.5, zorder=4, linewidths=0)
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            ax.hlines(mean, i-0.18, i+0.18, colors=C_LINE, linewidth=2.0, zorder=5)
            span = max(vals) - min(vals)
            ax.text(i, max(vals) + span*0.06,
                    f"{mean:.3f}±{std:.3f}", ha="center", fontsize=6.0, color=c)

        ax.set_xticks(positions); ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        all_flat = [v for vals in all_vals for v in vals]
        span = max(all_flat) - min(all_flat)
        ax.set_ylim(min(all_flat) - span*0.05, max(all_flat) + span*0.28)
        panel_label(ax, lbl)
        nc_style(ax)

    fig.suptitle(f"Forward Model Performance ({len(ridge_d['seeds'])}-Seed CV, n=1920)",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_forward_comparison")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Seed Stability (standalone)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    seeds_data = {
        "GNN":      [4.192, 4.548, 4.193, 4.439, 3.801],
        "Ridge":    [4.138, 4.289, 4.251, 4.468, 4.236],
        "LightGBM": [4.219, 4.354, 4.437, 4.476, 4.077],
        "BiLSTM":   [4.468, 4.353, 4.778, 5.006, 4.156],
    }
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for model, rmses in seeds_data.items():
        ax.plot(range(5), rmses, "o-", color=MODEL_COLORS[model],
                linewidth=1.6, markersize=4.5, label=model)
    ax.set_xlabel("Random Seed"); ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_xticks(range(5)); ax.set_xlim(-0.3, 4.3); ax.set_ylim(3.6, 5.3)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor='#BDBDBD')
    panel_label(ax, 'a'); nc_style(ax)
    plt.tight_layout()
    save_fig(fig, "results/fig_seed_stability")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1+2 COMBINED – Forward Model seed-wise stability
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1and2():
    with open("results/ridge_31seed_result.json") as f:
        ridge_d = json.load(f)
    with open("results/ml_31seed_result.json") as f:
        ml_d = json.load(f)
    with open("results/gnn_31seed_result.json") as f:
        gnn_d = json.load(f)
    with open("results/bilstm_31seed_result.json") as f:
        bl_d = json.load(f)

    # Build seed-wise data from actual JSONs (use first 31 seeds)
    def extract(seeds_list, key_map):
        return {k: [r[v] for r in seeds_list] for k, v in key_map.items()}

    key_map = {"RMSE": "RMSE", "R2": "R2", "Sp": "Spearman"}
    seeds_data = {
        "Ridge":    extract(ridge_d["seeds"],           key_map),
        "GNN":      extract(gnn_d["seeds"],             key_map),
        "BiLSTM":   extract(bl_d["seeds"],              key_map),
        "LightGBM": extract(ml_d["LightGBM"]["seeds"],  key_map),
        "XGBoost":  extract(ml_d["XGBoost"]["seeds"],   key_map),
        "SVR":      extract(ml_d["SVR"]["seeds"],        key_map),
    }
    models = list(seeds_data.keys())
    n_seeds = len(ridge_d["seeds"])
    s = list(range(n_seeds))

    # R² → supplementary; main: RMSE + Spearman only
    metrics = [
        ("RMSE", "RMSE (kcal/mol)", "a", (2, 8)),
        ("Sp",   "Spearman ρ",      "b", (0.5, 1.0)),
    ]

    # Layout constants
    PANEL = 2.8; L, R = 0.75, 0.25; T, B = 0.45, 0.65; HG = 0.70
    W = L + 2*PANEL + 1*HG + R
    H = T + PANEL + B
    fig = plt.figure(figsize=(W, H))

    axes = []
    for col in range(2):
        x0 = L + col*(PANEL+HG); y0 = B
        axes.append(fig.add_axes([x0/W, y0/H, PANEL/W, PANEL/H]))

    for ax, (key, ylabel, lbl, ylim) in zip(axes, metrics):
        for model in models:
            ax.plot(s, seeds_data[model][key], "o-",
                    color=MODEL_COLORS[model], linewidth=1.2,
                    markersize=3.0, alpha=0.85, label=model)
        ax.set_xlabel("Seed"); ax.set_ylabel(ylabel)
        ax.set_xticks(s[::5]); ax.set_xlim(-0.5, n_seeds - 0.5); ax.set_ylim(ylim)
        panel_label(ax, lbl); nc_style(ax)
        if lbl == "a":
            ax.legend(loc="upper right", framealpha=0.9, edgecolor='#BDBDBD', fontsize=7.0,
                      ncol=2)

    fig.text(0.5, 1.0, f"Forward Model Seed-wise Stability ({n_seeds}-Seed CV)",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             transform=fig.transFigure)
    save_fig(fig, "results/fig_forward_combined")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – Multi-target Reverse Design (grouped bar, standalone)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    with open("results/multitarget_reverse_result.json") as f:
        data = json.load(f)

    targets = [-30, -40]
    methods = ["GA", "BO"]
    cols    = [METHOD_COLORS[m] for m in methods]
    x = np.arange(len(targets)); w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    fig.subplots_adjust(wspace=0.40)

    panels = [
        ("Ridge_hit3", "Hit Rate (%)", (0, 120), "Ridge Hit Rate", "a", ".0f"),
        ("QED",        "QED",          (0, 0.95), "Drug-likeness (QED)", "b", ".2f"),
        ("novelty",    "Novelty (%)",  (0, 120), "Novelty", "c", ".0f"),
    ]
    for ax, (key, ylabel, ylim, title, lbl, fmt) in zip(axes, panels):
        for i, (m, c) in enumerate(zip(methods, cols)):
            vals = [data[str(t)][m][key] for t in targets]
            ax.bar(x + (i-0.5)*w, vals, width=w, color=c, alpha=0.85, label=m)
            for xi, v in zip(x, vals):
                ax.text(xi+(i-0.5)*w, v+ylim[1]*0.02,
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"dG={t}" for t in targets])
        ax.set_ylabel(ylabel); ax.set_ylim(ylim)
        ax.set_title(title, pad=4)
        panel_label(ax, lbl); nc_style(ax)
        ax.legend(framealpha=0.9, edgecolor='#BDBDBD')
        if key == "QED":
            ax.axhline(0.5, color=C_GRAY1, linestyle='--', linewidth=0.8, alpha=0.7)

    fig.suptitle("Reverse Design: GA vs. Bayesian Optimization (dG = −30, −40 kcal/mol)",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_multitarget_reverse")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 – Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    """Figure 4a: Ridge Feature Importance (standalone)"""
    with open("results/bo_feature_importance.json") as f:
        fi = json.load(f)

    enhancing = [x for x in fi["binding_enhancing_features"]
                 if not x["feature"].startswith("ECFP4")][:10]
    weakening  = [x for x in fi["binding_weakening_features"]
                  if not x["feature"].startswith("ECFP4")][:5]
    all_feats = [x["feature"] for x in enhancing] + [x["feature"] for x in weakening]
    all_imps  = [-abs(x["impact"]) for x in enhancing] + [x["impact"] for x in weakening]
    bar_cols  = [C_BEST] * len(enhancing) + [C_GRAY2] * len(weakening)
    idx = np.argsort(all_imps)
    feats_s = [all_feats[i] for i in idx]
    imps_s  = [all_imps[i]  for i in idx]
    cols_s  = [bar_cols[i]  for i in idx]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    y = np.arange(len(feats_s))
    ax.barh(y, imps_s, color=cols_s, alpha=0.85, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(feats_s, fontsize=8)
    ax.axvline(0, color=C_LINE, linewidth=0.7)
    ax.set_xlabel("Impact Score  (Ridge coef × Δmean)")
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_BEST,  label="Binding-enhancing"),
        mpatches.Patch(facecolor=C_GRAY2, label="Binding-weakening"),
    ], loc="lower right", framealpha=0.9, edgecolor='#BDBDBD')
    panel_label(ax, 'a'); nc_style(ax)
    fig.suptitle("Forward — Ridge Feature Importance",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_feature_importance")


def make_fig4b():
    """Figure 4b: BO Molecule Properties (panel a) + QED comparison (panel b)"""
    # Panel a: normalized physicochemical properties (excl. QED)
    props      = ["MolWt\n(/500)", "LogP\n(/5)", "SA\n(/5)", "Lipinski\n(%)"]
    bo_vals    = [187.8/500, 1.22/5, 1.09/5, 1.00]
    train_vals = [350/500,   2.10/5, 2.50/5, 0.95]
    # Panel b: QED distribution comparison
    qed_bo    = 0.690
    qed_train = 0.459

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.5),
                                      gridspec_kw={'width_ratios': [2.5, 1]})
    fig.subplots_adjust(wspace=0.45)

    # Panel a
    x3 = np.arange(len(props)); wb = 0.35
    ax_a.bar(x3-wb/2, train_vals, width=wb, color=C_GRAY3, alpha=0.9, label="Training set")
    ax_a.bar(x3+wb/2, bo_vals,    width=wb, color=C_BEST,  alpha=0.85, label="BO-generated")
    ax_a.set_xticks(x3); ax_a.set_xticklabels(props, fontsize=8)
    ax_a.set_ylabel("Normalized value"); ax_a.set_ylim(0, 1.18)
    ax_a.legend(framealpha=0.9, edgecolor='#BDBDBD', fontsize=8)
    panel_label(ax_a, 'a'); nc_style(ax_a)
    ax_a.set_title("Physicochemical Properties", pad=4)

    # Panel b: QED
    x_q = np.array([0, 1]); wb2 = 0.50
    ax_b.bar([0], [qed_train], width=wb2, color=C_GRAY3, alpha=0.9, label="Training set")
    ax_b.bar([1], [qed_bo],    width=wb2, color=C_BEST,  alpha=0.85, label="BO-generated")
    ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(["Train", "BO"], fontsize=8)
    ax_b.set_ylabel("QED"); ax_b.set_ylim(0, 1.0)
    ax_b.axhline(0.5, color=C_GRAY1, linestyle='--', linewidth=0.8, alpha=0.7)
    ax_b.text(0, qed_train + 0.03, f"{qed_train:.3f}", ha='center', fontsize=8)
    ax_b.text(1, qed_bo    + 0.03, f"{qed_bo:.3f}",    ha='center', fontsize=8, color=C_BEST)
    panel_label(ax_b, 'b'); nc_style(ax_b)
    ax_b.set_title("Drug-likeness (QED)", pad=4)

    fig.suptitle("Reverse — BO Molecule Physicochemical Properties",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_bo_properties")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3+5 COMBINED – Reverse Design GA vs BO at -30 and -40
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3and5():
    with open("results/multitarget_reverse_result.json") as f:
        data = json.load(f)

    five_seed_path = "results/multitarget_5seed_result.json"
    use_5seed = os.path.exists(five_seed_path)
    if use_5seed:
        with open(five_seed_path) as f:
            data5 = json.load(f)

    targets = [-30, -40]
    methods = ["GA", "BO"]
    colors  = [METHOD_COLORS["GA"], METHOD_COLORS["BO"]]

    # Layout: 2 rows × 3 cols
    PANEL = 2.6; L, R = 0.75, 0.30; T, B = 0.45, 0.65; HG, VG = 0.65, 0.75
    NROWS, NCOLS = 2, 3
    W = L + NCOLS*PANEL + (NCOLS-1)*HG + R
    H = T + NROWS*PANEL + (NROWS-1)*VG + B
    fig = plt.figure(figsize=(W, H))

    def add_ax(col, row):
        x0 = L + col*(PANEL+HG); y0 = B + (NROWS-1-row)*(PANEL+VG)
        return fig.add_axes([x0/W, y0/H, PANEL/W, PANEL/H])

    panel_iter = iter("abcdef")
    w_bar = 0.40

    for row, (key, ylabel, ylim, title, fmt) in enumerate([
        ("Ridge_hit3", "Hit Rate (%)", (0, 120), "Ridge Hit Rate", ".0f"),
        ("QED",        "QED",          (0, 1.0),  "Drug-likeness (QED)", ".3f"),
    ]):
        for col, t in enumerate(targets):
            ax = add_ax(col, row)
            lbl = next(panel_iter)
            for i, (m, c) in enumerate(zip(methods, colors)):
                val = data[str(t)][m][key]
                err = (data5[str(t)][m]["std"][key] if use_5seed
                       and key in data5[str(t)][m].get("std", {}) else 0)
                ax.bar(i, val, width=w_bar, color=c, alpha=0.85,
                       yerr=err if use_5seed else None,
                       capsize=3.5, error_kw={"linewidth": 1.0, "ecolor": C_LINE})
                ax.text(i, val+(err if use_5seed else 0)+ylim[1]*0.03,
                        f"{val:{fmt}}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(methods)
            ax.set_ylabel(ylabel if col == 0 else ""); ax.set_ylim(ylim)
            ax.set_title(f"{title}\ndG = {t} kcal/mol", pad=4)
            panel_label(ax, lbl); nc_style(ax)
            if key == "QED":
                ax.axhline(0.5, color=C_GRAY1, linestyle='--', linewidth=0.8, alpha=0.7)

    # Novelty panel: right column, spans both rows
    ax_nov = add_ax(2, 0)
    ax_nov.set_position([
        (L + 2*(PANEL+HG))/W,
        B/H,
        PANEL/W,
        (2*PANEL + VG)/H
    ])
    lbl = next(panel_iter)
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [data[str(t)][m]["novelty"] for t in targets]
        ax_nov.bar(np.arange(2)+(i-0.5)*0.35, vals, width=0.35, color=c, alpha=0.85, label=m)
        for xi, v in zip(range(2), vals):
            ax_nov.text(xi+(i-0.5)*0.35, v+2, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax_nov.set_xticks([0, 1]); ax_nov.set_xticklabels([f"dG={t}" for t in targets])
    ax_nov.set_ylabel("Novelty (%)"); ax_nov.set_ylim(0, 120)
    ax_nov.set_title("Novelty vs. Training Set", pad=4)
    ax_nov.legend(framealpha=0.9, edgecolor='#BDBDBD')
    panel_label(ax_nov, lbl); nc_style(ax_nov)

    suffix = " (5-Seed ± SD)" if use_5seed else ""
    fig.text(W/2/W, 1.0, f"Reverse Design: GA vs. Bayesian Optimization{suffix}",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             transform=fig.transFigure)
    save_fig(fig, "results/fig_reverse_combined")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 – Reverse Summary (standalone)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig5():
    five_seed_path = "results/multitarget_5seed_result.json"
    single_path    = "results/multitarget_reverse_result.json"
    use_5seed = os.path.exists(five_seed_path)

    if use_5seed:
        with open(five_seed_path) as f:
            d = json.load(f)
        get_mean = lambda t, m, k: d[t][m]["mean"][k]
        get_std  = lambda t, m, k: d[t][m]["std"][k]
    else:
        with open(single_path) as f:
            d = json.load(f)
        get_mean = lambda t, m, k: d[t][m][k]
        get_std  = lambda t, m, k: 0.0

    methods = ["GA", "BO"]
    colors  = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.2))
    fig.subplots_adjust(wspace=0.42)

    for ax, (key, ylabel, ylim, lbl, fmt) in zip(axes, [
        ("Ridge_hit3", "Ridge Hit Rate (%)", (0, 120), "a", ".1f"),
        ("QED",        "QED",                (0, 0.9),  "b", ".3f"),
    ]):
        for i, (m, c) in enumerate(zip(methods, colors)):
            mean = get_mean("-30", m, key)
            std  = get_std("-30", m, key)
            ax.bar(i, mean, width=0.5, color=c, alpha=0.85,
                   yerr=std if use_5seed else None,
                   capsize=3.5, error_kw={"linewidth": 1.0, "ecolor": C_LINE})
            ax.text(i, mean+(std if use_5seed else 0)+ylim[1]*0.03,
                    f"{mean:{fmt}}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel); ax.set_ylim(ylim)
        panel_label(ax, lbl); nc_style(ax)
        if key == "QED":
            ax.axhline(0.5, color=C_GRAY1, linestyle='--', linewidth=0.8, alpha=0.7)

    suffix = " (5-Seed Mean ± SD)" if use_5seed else " (single run)"
    fig.suptitle(f"GA vs. Bayesian Optimization at dG = −30 kcal/mol{suffix}",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_reverse_summary")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 – Supplementary: All Models 5-Seed
# ══════════════════════════════════════════════════════════════════════════════
def make_fig6():
    # Check for updated ChemBERTa 5-seed result
    cb_v2_path = "results/5seed_chemberta_v2_result.json"
    if os.path.exists(cb_v2_path):
        with open(cb_v2_path) as f:
            cb = json.load(f)
        cb_rmse_m = cb["mean"]["RMSE"]; cb_rmse_s = cb["std"]["RMSE"]
        cb_r2_m   = cb["mean"]["R2"];   cb_r2_s   = cb["std"]["R2"]
        cb_sp_m   = cb["mean"]["Spearman"]; cb_sp_s = cb["std"]["Spearman"]
        cb_marker = ""
    else:
        cb_rmse_m, cb_rmse_s = 5.018, 0.0
        cb_r2_m,   cb_r2_s   = 0.656, 0.0
        cb_sp_m,   cb_sp_s   = 0.826, 0.0
        cb_marker = "*"

    models = ["GNN", "Ridge", "LightGBM", "XGBoost", "BiLSTM", "ChemBERTa", "SVR"]
    RMSE_m = [4.234, 4.276, 4.313, 4.548, 4.552, cb_rmse_m, 5.379]
    RMSE_s = [0.258, 0.108, 0.147, 0.266, 0.304, cb_rmse_s, 0.220]
    R2_m   = [0.712, 0.708, 0.703, 0.693, 0.669, cb_r2_m,   0.570]
    R2_s   = [0.043, 0.017, 0.024, 0.026, 0.037, cb_r2_s,   0.029]
    Sp_m   = [0.850, 0.849, 0.838, 0.848, 0.827, cb_sp_m,   0.804]
    Sp_s   = [0.019, 0.009, 0.009, 0.009, 0.021, cb_sp_s,   0.009]

    colors_m = [MODEL_COLORS.get(m, C_GRAY2) for m in models]

    data = [
        ("RMSE (↓)",      RMSE_m, RMSE_s, (0, 7.0)),
        ("R² (↑)",         R2_m,   R2_s,   (0, 1.0)),
        ("Spearman ρ (↑)", Sp_m,   Sp_s,   (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    fig.subplots_adjust(wspace=0.40)
    lbls = iter("abc")

    for ax, (metric, vals, errs, ylim) in zip(axes, data):
        ax.bar(range(len(models)), vals, color=colors_m, alpha=0.88,
               yerr=errs, capsize=3, error_kw={"linewidth": 0.9, "ecolor": C_LINE})
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=38, ha="right")
        ax.set_ylim(ylim); ax.set_ylabel(metric)
        # asterisk for ChemBERTa single run
        if cb_marker:
            cb_i = models.index("ChemBERTa")
            ax.text(cb_i, vals[cb_i] + (ylim[1]-ylim[0])*0.015, "*",
                    ha="center", fontsize=9, color=C_GRAY1)
        panel_label(ax, next(lbls)); nc_style(ax)

    if cb_marker:
        axes[0].text(0.01, 0.01, "* single run", transform=axes[0].transAxes,
                     fontsize=6.5, color=C_GRAY1, style="italic")
    axes[0].legend(handles=[
        mpatches.Patch(facecolor=C_BEST,  label="Ridge (selected)"),
        mpatches.Patch(facecolor=C_BLUE2, label="GNN"),
        mpatches.Patch(facecolor=C_TEAL1, label="LightGBM / XGBoost"),
        mpatches.Patch(facecolor=C_GRAY2, label="BiLSTM / ChemBERTa / SVR"),
    ], loc="upper right", fontsize=6.5, framealpha=0.9, edgecolor='#BDBDBD')

    fig.suptitle("All-Model 5-Seed Performance Summary (Mean ± SD)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/fig_5seed_metrics")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 – Reverse Design 5-method comparison (when available)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig7():
    """Figure 2 (paper): Reverse Design — violin + scatter style (like Figure 1).
    Row 0: Independent hit rate (5 seeds × 3 validators = 15 dots) per target
    Row 1: QED | SA Score | Lipinski (5 seeds × 2 targets = 10 dots each)"""
    result_path = "results/reverse_5method_5seed_result.json"
    if not os.path.exists(result_path):
        print("Skipped: results/reverse_5method_5seed_result.json not found")
        return

    with open(result_path) as f:
        data = json.load(f)

    targets = [-30, -40]
    methods = ["GA", "BO", "ScaffoldHop", "Fragment", "Retrieval"]
    colors  = [METHOD_COLORS.get(m, C_TEAL2) for m in methods]
    VALIDATORS = ["LGB_hit3", "GNN_hit3", "BiLSTM_hit3"]

    def violin_scatter(ax, all_vals, colors, ylabel, title, ylim=None, refline=None):
        """Draw violin + scatter (style matching Figure 1)."""
        positions = list(range(len(methods)))
        # Filter out empty/degenerate distributions
        clean = [v if len(set(v)) > 1 else v + [v[0]+1e-6] for v in all_vals]
        vp = ax.violinplot(clean, positions=positions, widths=0.5,
                           showmeans=False, showmedians=False, showextrema=False)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.30)
            body.set_edgecolor(c); body.set_linewidth(0.8)
        for i, (vals, c) in enumerate(zip(all_vals, colors)):
            ax.scatter([i]*len(vals), vals, color=c, s=7, alpha=0.55, zorder=4, linewidths=0)
            mean = float(np.mean(vals))
            ax.hlines(mean, i-0.18, i+0.18, colors=C_LINE, linewidth=2.0, zorder=5)
            std = float(np.std(vals))
            ax.text(i, max(vals) + (max(vals)-min(vals)+1)*0.06,
                    f"{mean:.1f}±{std:.1f}", ha="center", fontsize=5.5, color=c)
        ax.set_xticks(positions); ax.set_xticklabels(methods, rotation=28, ha='right', fontsize=7.5)
        ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(ylim)
        if refline is not None:
            ax.axhline(refline, color=C_GRAY1, ls='--', lw=0.8, alpha=0.7)
        ax.set_title(title, pad=4, fontsize=8.5)
        nc_style(ax)

    # ── Layout: 2 rows ────────────────────────────────────────────────────────
    # Row 0 (2 panels): Hit rate per target
    # Row 1 (3 panels): QED | SA Score | Lipinski
    PANEL_W = 3.0; PANEL_H = 2.7; PANEL_W3 = 2.3
    L, R = 0.75, 0.25; T, B = 0.45, 0.55; HG = 0.65; VG = 0.65; HG3 = 0.45

    W  = L + 2*PANEL_W + HG + R
    H  = T + 2*PANEL_H + VG + B
    fig = plt.figure(figsize=(W, H))

    panel_iter = iter("abcdef")

    # ── Row 0: Hit rate violin (15 dots = 5 seeds × 3 validators) ────────────
    for col, t in enumerate(targets):
        x0 = L + col*(PANEL_W+HG)
        y0 = B + PANEL_H + VG
        ax = fig.add_axes([x0/W, y0/H, PANEL_W/W, PANEL_H/H])
        lbl = next(panel_iter)
        # 15 data points per method: 5 seeds × 3 validators
        all_vals = []
        for m in methods:
            pts = []
            for seed_r in data[str(t)][m]["seeds"]:
                for v in VALIDATORS:
                    pts.append(seed_r.get(v, 0.0))
            all_vals.append(pts)
        violin_scatter(ax, all_vals, colors,
                       "Hit Rate (%)",
                       f"Independent Hit Rate  (dG={t} kcal/mol)\n"
                       r"5 seeds × LGB·GNN·BiLSTM (n=15/method)",
                       ylim=(0, 110))
        panel_label(ax, lbl)

    # ── Row 1: QED | SA Score | Lipinski (10 dots = 5 seeds × 2 targets) ─────
    total_bottom_w = 3*PANEL_W3 + 2*HG3
    x0_start = L + (2*PANEL_W + HG - total_bottom_w) / 2
    for col, (key, ylabel, ylim, refline, title_str) in enumerate([
        ("QED",      "QED",                      (0.0, 1.05), 0.5,  "Drug-likeness (QED)"),
        ("SA_score", "SA Score",                  (1.0, 6.5),  3.0,  "Synthetic Accessibility"),
        ("Lipinski", "Lipinski Ro5 (%)",           (0, 115),   100.0, "Lipinski Compliance"),
    ]):
        x0 = x0_start + col*(PANEL_W3+HG3)
        ax = fig.add_axes([x0/W, B/H, PANEL_W3/W, PANEL_H/H])
        lbl = next(panel_iter)
        # 10 data points: 5 seeds × 2 targets
        all_vals = []
        for m in methods:
            pts = []
            for t2 in targets:
                for seed_r in data[str(t2)][m]["seeds"]:
                    pts.append(seed_r.get(key, 0.0))
            all_vals.append(pts)
        violin_scatter(ax, all_vals, colors, ylabel, title_str, ylim=ylim, refline=refline)
        panel_label(ax, lbl)

    fig.text(0.5, 1.0,
             "Reverse Design: 5-Method Comparison  (n=50 molecules/seed)",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             transform=fig.transFigure)
    save_fig(fig, "results/fig_reverse_design")


def make_fig_forward():
    """Figure 1 (paper): Forward model benchmarking — violin + seed stability combined."""
    with open("results/ridge_31seed_result.json") as f:
        ridge_d = json.load(f)
    with open("results/ml_31seed_result.json") as f:
        ml_d = json.load(f)
    with open("results/gnn_31seed_result.json") as f:
        gnn_d = json.load(f)
    with open("results/bilstm_31seed_result.json") as f:
        bl_d = json.load(f)

    model_data = {
        "Ridge":    ridge_d["seeds"],
        "GNN":      gnn_d["seeds"],
        "BiLSTM":   bl_d["seeds"],
        "LightGBM": ml_d["LightGBM"]["seeds"],
        "XGBoost":  ml_d["XGBoost"]["seeds"],
        "SVR":      ml_d["SVR"]["seeds"],
    }
    models = list(model_data.keys())
    colors_v = [MODEL_COLORS.get(m, C_GRAY2) for m in models]
    n_seeds = len(ridge_d["seeds"])
    s = list(range(n_seeds))

    # Layout: top row = 2 violin panels (RMSE, Spearman), bottom = seed-wise line (full width)
    PANEL_W = 3.2; PANEL_H = 2.8; LINE_H = 2.4
    L, R = 0.75, 0.25; T, B = 0.45, 0.55; HG = 0.70; VG = 0.65

    W = L + 2*PANEL_W + HG + R
    H = T + PANEL_H + VG + LINE_H + B
    fig = plt.figure(figsize=(W, H))

    def add_violin_ax(col):
        x0 = L + col*(PANEL_W+HG)
        y0 = B + LINE_H + VG
        return fig.add_axes([x0/W, y0/H, PANEL_W/W, PANEL_H/H])

    def add_line_ax(col):
        x0 = L + col*(PANEL_W+HG)
        y0 = B
        return fig.add_axes([x0/W, y0/H, PANEL_W/W, LINE_H/H])

    # Panel a: RMSE violin
    ax_rmse = add_violin_ax(0)
    all_rmse = [[r["RMSE"] for r in model_data[m]] for m in models]
    vp = ax_rmse.violinplot(all_rmse, positions=range(len(models)), widths=0.45,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], colors_v):
        body.set_facecolor(c); body.set_alpha(0.35)
        body.set_edgecolor(c); body.set_linewidth(0.8)
    for i, (vals, c) in enumerate(zip(all_rmse, colors_v)):
        ax_rmse.scatter([i]*len(vals), vals, color=c, s=5, alpha=0.45, zorder=4, linewidths=0)
        mean = float(np.mean(vals)); std = float(np.std(vals))
        ax_rmse.hlines(mean, i-0.18, i+0.18, colors=C_LINE, linewidth=2.0, zorder=5)
        span = max(vals)-min(vals)
        ax_rmse.text(i, max(vals)+span*0.07, f"{mean:.2f}±{std:.2f}",
                     ha="center", fontsize=5.8, color=c)
    ax_rmse.set_xticks(range(len(models)))
    ax_rmse.set_xticklabels(models, rotation=22, ha='right')
    ax_rmse.set_ylabel("RMSE (kcal/mol)")
    all_flat = [v for vals in all_rmse for v in vals]
    span = max(all_flat)-min(all_flat)
    ax_rmse.set_ylim(min(all_flat)-span*0.05, max(all_flat)+span*0.30)
    panel_label(ax_rmse, 'a'); nc_style(ax_rmse)

    # Panel b: Spearman violin
    ax_sp = add_violin_ax(1)
    all_sp = [[r["Spearman"] for r in model_data[m]] for m in models]
    vp2 = ax_sp.violinplot(all_sp, positions=range(len(models)), widths=0.45,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, c in zip(vp2["bodies"], colors_v):
        body.set_facecolor(c); body.set_alpha(0.35)
        body.set_edgecolor(c); body.set_linewidth(0.8)
    for i, (vals, c) in enumerate(zip(all_sp, colors_v)):
        ax_sp.scatter([i]*len(vals), vals, color=c, s=5, alpha=0.45, zorder=4, linewidths=0)
        mean = float(np.mean(vals)); std = float(np.std(vals))
        ax_sp.hlines(mean, i-0.18, i+0.18, colors=C_LINE, linewidth=2.0, zorder=5)
        span = max(vals)-min(vals)
        ax_sp.text(i, max(vals)+span*0.07, f"{mean:.3f}±{std:.3f}",
                   ha="center", fontsize=5.8, color=c)
    ax_sp.set_xticks(range(len(models)))
    ax_sp.set_xticklabels(models, rotation=22, ha='right')
    ax_sp.set_ylabel("Spearman ρ")
    all_flat2 = [v for vals in all_sp for v in vals]
    span2 = max(all_flat2)-min(all_flat2)
    ax_sp.set_ylim(min(all_flat2)-span2*0.05, max(all_flat2)+span2*0.30)
    panel_label(ax_sp, 'b'); nc_style(ax_sp)

    # Panel c: Seed-wise RMSE stability (full width bottom)
    ax_line = fig.add_axes([L/W, B/H, (2*PANEL_W+HG)/W, LINE_H/H])
    for model in models:
        rmse_vals = [r["RMSE"] for r in model_data[model]]
        ax_line.plot(s, rmse_vals, "o-", color=MODEL_COLORS.get(model, C_GRAY2),
                     linewidth=1.2, markersize=2.8, alpha=0.85, label=model)
    ax_line.set_xlabel("Seed"); ax_line.set_ylabel("RMSE (kcal/mol)")
    ax_line.set_xticks(s[::5]); ax_line.set_xlim(-0.5, n_seeds-0.5)
    ax_line.set_ylim(2, 8)
    ax_line.legend(ncol=6, fontsize=7.5, framealpha=0.9, edgecolor='#BDBDBD',
                   loc='upper center', bbox_to_anchor=(0.5, 1.20))
    panel_label(ax_line, 'c'); nc_style(ax_line)

    fig.text(0.5, 1.0, f"Forward MMGBSA Prediction: 6 Models ({n_seeds}-Seed CV, n=1920)",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             transform=fig.transFigure)
    save_fig(fig, "results/fig1_forward")


def make_fig3_analysis():
    """Figure 3 (paper): Feature importance + BO properties combined."""
    with open("results/bo_feature_importance.json") as f:
        fi = json.load(f)

    enhancing = [x for x in fi["binding_enhancing_features"]
                 if not x["feature"].startswith("ECFP4")][:10]
    weakening  = [x for x in fi["binding_weakening_features"]
                  if not x["feature"].startswith("ECFP4")][:5]
    all_feats = [x["feature"] for x in enhancing] + [x["feature"] for x in weakening]
    all_imps  = [-abs(x["impact"]) for x in enhancing] + [x["impact"] for x in weakening]
    bar_cols  = [C_BEST]*len(enhancing) + [C_GRAY2]*len(weakening)
    idx = np.argsort(all_imps)

    props      = ["MolWt\n(/500)", "LogP\n(/5)", "SA\n(/5)", "Lipinski\n(%)"]
    bo_vals    = [187.8/500, 1.22/5, 1.09/5, 1.00]
    train_vals = [350/500,   2.10/5, 2.50/5, 0.95]

    # Layout: panel a (feature importance, tall left) | panel b (mol props right)
    # QED removed — covered in Figure 2 (reverse design)
    PANEL_W = 3.8; PANEL_H_A = 4.5
    PANEL_W_B = 3.0; PANEL_H_B = 2.8
    L, R = 0.75, 0.25; T, B = 0.45, 0.55; HG = 0.70

    W = L + PANEL_W + HG + PANEL_W_B + R
    H = T + PANEL_H_A + B
    fig = plt.figure(figsize=(W, H))

    # Panel a: feature importance
    ax_a = fig.add_axes([L/W, B/H, PANEL_W/W, PANEL_H_A/H])
    y = np.arange(len(idx))
    ax_a.barh(y, [all_imps[i] for i in idx], color=[bar_cols[i] for i in idx],
              alpha=0.85, height=0.6)
    ax_a.set_yticks(y); ax_a.set_yticklabels([all_feats[i] for i in idx], fontsize=7.5)
    ax_a.axvline(0, color=C_LINE, linewidth=0.7)
    ax_a.set_xlabel("Impact  (Ridge coef × Δmean)")
    ax_a.legend(handles=[mpatches.Patch(facecolor=C_BEST, label="Binding-enhancing"),
                         mpatches.Patch(facecolor=C_GRAY2, label="Binding-weakening")],
                loc="lower right", framealpha=0.9, edgecolor='#BDBDBD')
    panel_label(ax_a, 'a'); nc_style(ax_a)
    ax_a.set_title("Ridge Feature Importance", pad=4)

    # Panel b: normalized properties (BO vs Training)
    x_l = L + PANEL_W + HG
    y_center = B + (PANEL_H_A - PANEL_H_B) / 2
    ax_b = fig.add_axes([x_l/W, y_center/H, PANEL_W_B/W, PANEL_H_B/H])
    x3 = np.arange(len(props)); wb = 0.35
    ax_b.bar(x3-wb/2, train_vals, width=wb, color=C_GRAY3, alpha=0.9, label="Training set")
    ax_b.bar(x3+wb/2, bo_vals,    width=wb, color=C_BEST,  alpha=0.85, label="BO-generated")
    ax_b.set_xticks(x3); ax_b.set_xticklabels(props, fontsize=8.5)
    ax_b.set_ylabel("Normalized value"); ax_b.set_ylim(0, 1.20)
    ax_b.legend(framealpha=0.9, edgecolor='#BDBDBD', fontsize=8.5)
    panel_label(ax_b, 'b'); nc_style(ax_b)
    ax_b.set_title("BO-Generated vs. Training Set\nMolecular Property Profile", pad=4)

    fig.suptitle("Feature Analysis & Molecular Properties (Ridge Feature Importance · BO vs. Training)",
                 fontsize=10, fontweight="bold", y=1.02)
    save_fig(fig, "results/fig3_analysis")


def make_supp1_allmodels():
    """Supp 1: All models R² + RMSE + Spearman (already fig_5seed_metrics, keep as-is)."""
    make_fig6()


def make_supp2_scatter():
    """Supp 2: Predicted vs Actual scatter plot (Ridge, seed 0)."""
    scatter_path = "results/ridge_seed0_preds.json"
    if not os.path.exists(scatter_path):
        print("Skipped: ridge_seed0_preds.json not found")
        return
    import json as _json
    with open(scatter_path) as f:
        d = _json.load(f)
    y_true = np.array(d["y_true"])
    y_pred = np.array(d["y_pred"])

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_squared_error

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    pcc  = float(pearsonr(y_true, y_pred)[0])
    rho  = float(spearmanr(y_true, y_pred)[0])

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(y_true, y_pred, color=C_BEST, s=12, alpha=0.5, linewidths=0)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], '--', color=C_LINE, linewidth=0.9, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual MMGBSA dG Bind (kcal/mol)")
    ax.set_ylabel("Predicted (kcal/mol)")
    ax.text(0.05, 0.95, f"RMSE={rmse:.2f}\nR²={r2:.3f}\nρ={rho:.3f}\nr={pcc:.3f}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BDBDBD'))
    panel_label(ax, 'a'); nc_style(ax)
    fig.suptitle("Supp. Fig. 2 | Ridge Predicted vs. Actual (Seed 0, n=288)",
                 fontsize=9, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, "results/supp2_scatter")


def make_supp3_properties():
    """Supp 3: Training set molecular property distributions."""
    import sys as _sys
    _sys.path.insert(0, '.')
    from data_utils import load_data
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED

    df = load_data()
    mols = [Chem.MolFromSmiles(s) for s in df["canonical_smiles"]]
    mols = [m for m in mols if m is not None]

    mw   = [Descriptors.MolWt(m) for m in mols]
    logp = [Descriptors.MolLogP(m) for m in mols]
    tpsa = [Descriptors.TPSA(m) for m in mols]
    hbd  = [Descriptors.NumHDonors(m) for m in mols]
    hba  = [Descriptors.NumHAcceptors(m) for m in mols]
    qed_vals = [QED.qed(m) for m in mols]
    target = df["MMGBSA dG Bind"].values

    props = [
        (mw,      "Molecular Weight (Da)",    "a", (0, 800)),
        (logp,    "LogP",                     "b", (-5, 10)),
        (tpsa,    "TPSA (Å²)",               "c", (0, 300)),
        (hbd,     "H-Bond Donors",            "d", (0, 12)),
        (qed_vals,"QED Drug-likeness",        "e", (0, 1.0)),
        (target,  "MMGBSA dG Bind (kcal/mol)","f", (-55, 15)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    fig.subplots_adjust(hspace=0.50, wspace=0.38)

    for ax, (vals, xlabel, lbl, xlim) in zip(axes.flat, props):
        ax.hist(vals, bins=35, color=C_BEST, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.set_xlim(xlim)
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.axvline(mu, color=C_LINE, lw=1.2, ls='--')
        ax.text(0.97, 0.95, f"μ={mu:.1f}\nσ={sd:.1f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
        panel_label(ax, lbl); nc_style(ax)

    fig.suptitle(f"Supp. Fig. 3 | Training Set Molecular Property Distributions (n={len(mols)})",
                 fontsize=10, fontweight='bold', y=1.02)
    save_fig(fig, "results/supp3_properties")


def make_supp4_hitrate():
    """Supp 4: Ridge hit rate vs LGB hit rate side-by-side (circular vs independent)."""
    result_path = "results/reverse_5method_5seed_result.json"
    if not os.path.exists(result_path):
        print("Skipped")
        return
    with open(result_path) as f:
        data = json.load(f)

    targets = [-30, -40]
    methods = ["GA", "BO", "ScaffoldHop", "Fragment", "Retrieval"]
    colors  = [METHOD_COLORS.get(m, C_TEAL2) for m in methods]
    x = np.arange(len(methods)); w = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    fig.subplots_adjust(hspace=0.52, wspace=0.38)

    pairs = [
        ("Ridge_hit3", "Ridge Hit Rate (%)"),
        ("LGB_hit3",   "LightGBM Hit Rate (%)"),
    ]
    for col, (key, ylabel) in enumerate(pairs):
        for row, t in enumerate(targets):
            ax = axes[row, col]
            vals = [data[str(t)][m]["mean"][key] for m in methods]
            errs = [data[str(t)][m]["std"][key]  for m in methods]
            ax.bar(x, vals, width=w, color=colors, alpha=0.85,
                   yerr=errs, capsize=3, error_kw={"linewidth":0.9, "ecolor":C_LINE})
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=28, ha='right', fontsize=7.5)
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_ylim(0, 115)
            ax.set_title(f"{ylabel.split(' (')[0]}  dG={t}", pad=4, fontsize=8.5)
            lbl = chr(ord('a') + row*2 + col)
            panel_label(ax, lbl); nc_style(ax)
            if col == 0:
                ax.axhline(100, color=C_GRAY1, ls='--', lw=0.8, alpha=0.6)

    fig.suptitle("Supp. Fig. 4 | Ridge Hit Rate (Circular) vs. LightGBM Hit Rate (Independent)\n"
                 "Ridge = fitness function → always 100%. LGB provides unbiased validation.",
                 fontsize=9, fontweight='bold', y=1.02)
    save_fig(fig, "results/supp4_hitrate")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # Legacy helpers (kept for supp)
    make_fig1()
    make_fig2()
    make_fig1and2()
    make_fig3()
    make_fig4()
    make_fig4b()
    make_fig5()
    make_fig3and5()
    make_fig6()
    make_fig7()
    # Paper-structured main figures
    make_fig_forward()
    make_fig3_analysis()
    # Supplementary
    make_supp2_scatter()
    make_supp3_properties()
    make_supp4_hitrate()
    print("\nAll figures regenerated.")
