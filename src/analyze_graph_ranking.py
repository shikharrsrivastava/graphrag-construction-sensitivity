"""Phase C analysis: exploratory tables and figures from saved per-question results.

Reads only results/graph_ranking_v1/{hotpot498,musique500}/question_results.jsonl and
the pre-registered headline tables. Everything written here except the copied
headline table is EXPLORATORY (see PREREGISTRATION_graph_ranking.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common import ROOT  # noqa: E402
from src.evaluate_dense_graph import bootstrap_mean_ci, dump_csv, holm_adjust  # noqa: E402
from src.evaluate_graph_ranking import STAT_SEED, paired_diff  # noqa: E402

DATASETS = {"hotpot498": "HotpotQA (n=498)", "musique500": "MuSiQue (n=500)", "musique500_rep": "MuSiQue replication (n=500)"}
GRAPH_CONFIGS = ["ppr_entity", "ppr_chunk", "ppr_entity_loo", "ppr_chunk_loo",
                 "ppr_entity_lambda0.25", "ppr_entity_lambda0.75", "ppr_chunk_lambda0.25", "ppr_chunk_lambda0.75"]
MAIN = ["ppr_entity", "ppr_chunk"]
# Validated categorical slots 1-3 (dataviz reference palette, light) + neutral gray for the lexical baseline.
COLOR = {"bm25": "#8a8985", "vanilla_bge": "#2a78d6", "entity": "#eb6834", "chunk": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(run_dir: Path) -> list[dict]:
    with (run_dir / "question_results.jsonl").open(encoding="utf-8") as h:
        return [json.loads(line) for line in h]


def index(rows):
    return {(r["question_id"], r["config"]): r for r in rows}


def group_of(row, dataset):
    return row["question_type"]  # bridge/comparison for HotpotQA, 2hop/3hop/4hop for MuSiQue


def by_group_contrasts(rows, dataset):
    out = []
    groups = sorted({group_of(r, dataset) for r in rows})
    for g in groups:
        sub = [r for r in rows if group_of(r, dataset) == g]
        for cfg in MAIN + ["ppr_entity_loo", "ppr_chunk_loo"]:
            for m in ("complete_support_at_5", "recall_at_5"):
                d = paired_diff(sub, cfg, "vanilla_bge", m)
                out.append({"dataset": dataset, "question_group": g, **d, "status": "EXPLORATORY (unadjusted p)"})
    return out


def loo_contrasts(rows, dataset):
    tests = [paired_diff(rows, c, "vanilla_bge", m) for c in ("ppr_entity_loo", "ppr_chunk_loo")
             for m in ("complete_support_at_5", "recall_at_5")]
    for t, adj in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        t.update(dataset=dataset, holm_adjusted_p=adj, holm_family="4 exploratory LOO tests within dataset",
                 status="EXPLORATORY (added after smoke test)")
    return tests


def lambda_sensitivity(rows, dataset):
    out = []
    for g in ("entity", "chunk"):
        for lam, cfg in ((0.25, f"ppr_{g}_lambda0.25"), (0.5, f"ppr_{g}"), (0.75, f"ppr_{g}_lambda0.75")):
            for m in ("complete_support_at_5", "recall_at_5"):
                d = paired_diff(rows, cfg, "vanilla_bge", m)
                out.append({"dataset": dataset, "graph": g, "lambda": lam, **d,
                            "status": "preregistered" if lam == 0.5 else "EXPLORATORY (unadjusted p)"})
    return out


def top5_changes(rows, dataset):
    ix = index(rows)
    qids = sorted({r["question_id"] for r in rows})
    out = []
    for cfg in GRAPH_CONFIGS:
        changed = [q for q in qids if set(ix[(q, cfg)]["ranked_passage_ids"][:5]) != set(ix[(q, "vanilla_bge")]["ranked_passage_ids"][:5])]
        order_changed = sum(ix[(q, cfg)]["ranked_passage_ids"] != ix[(q, "vanilla_bge")]["ranked_passage_ids"] for q in qids)
        rec = {"dataset": dataset, "config": cfg, "n_questions": len(qids), "full_ranking_changed": order_changed,
               "top5_set_changed": len(changed)}
        for m in ("complete_support_at_5", "recall_at_5"):
            diffs = [ix[(q, cfg)][m] - ix[(q, "vanilla_bge")][m] for q in changed]
            rec[f"{m}_improved"] = sum(d > 0 for d in diffs)
            rec[f"{m}_worse"] = sum(d < 0 for d in diffs)
            rec[f"{m}_unchanged"] = sum(d == 0 for d in diffs)
        rec["status"] = "EXPLORATORY"
        out.append(rec)
    return out


def gold_rank_movement(rows, dataset):
    """Per gold passage: rank under config minus rank under BGE (negative = moved up)."""
    ix = index(rows)
    qids = sorted({r["question_id"] for r in rows})
    summary, raw = [], {}
    for cfg in GRAPH_CONFIGS:
        moves = []
        for q in qids:
            b, c = ix[(q, "vanilla_bge")]["gold_ranks"], ix[(q, cfg)]["gold_ranks"]
            moves += [c[pid] - b[pid] for pid in b]
        moves = np.array(moves)
        raw[cfg] = moves
        crossed_in = crossed_out = 0
        for q in qids:
            b, c = ix[(q, "vanilla_bge")]["gold_ranks"], ix[(q, cfg)]["gold_ranks"]
            crossed_in += sum(b[p] > 5 >= c[p] for p in b)
            crossed_out += sum(c[p] > 5 >= b[p] for p in b)
        summary.append({"dataset": dataset, "config": cfg, "n_gold_passages": len(moves),
                        "mean_rank_change": float(moves.mean()), "moved_up": int((moves < 0).sum()),
                        "moved_down": int((moves > 0).sum()), "unchanged": int((moves == 0).sum()),
                        "entered_top5": crossed_in, "left_top5": crossed_out, "status": "EXPLORATORY"})
    return summary, raw


def conditional_on_bge(rows, dataset):
    ix = index(rows)
    qids = sorted({r["question_id"] for r in rows})
    out = []
    for stratum in (0, 1):
        sq = {q for q in qids if ix[(q, "vanilla_bge")]["complete_support_at_5"] == stratum}
        sub = [r for r in rows if r["question_id"] in sq]
        for cfg in MAIN + ["ppr_entity_loo", "ppr_chunk_loo"]:
            d = paired_diff(sub, cfg, "vanilla_bge", "complete_support_at_5")
            out.append({"dataset": dataset, "bge_complete_support_at_5": stratum, **d, "status": "EXPLORATORY (unadjusted p)"})
    return out


def headroom(rows, dataset):
    out = []
    for m in ("recall_at_5", "complete_support_at_5", "recall_at_10", "complete_support_at_10"):
        vals = [r[m] for r in rows if r["config"] == "vanilla_bge"]
        mean, lo, hi = bootstrap_mean_ci(vals, STAT_SEED + sum(map(ord, f"headroom:{dataset}:{m}")))
        out.append({"dataset": dataset, "config": "vanilla_bge", "metric": m, "mean": mean, "ci95_low": lo, "ci95_high": hi,
                    "context_count": rows[0]["context_count"], "status": "EXPLORATORY (headroom check)"})
    return out


def method_cis(rows, dataset, configs):
    out = {}
    for cfg in configs:
        vals = [r["complete_support_at_5"] for r in rows if r["config"] == cfg]
        out[cfg] = bootstrap_mean_ci(vals, STAT_SEED + sum(map(ord, f"fig:{dataset}:{cfg}")))
    return out


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_complete_support(all_rows, path):
    configs = ["bm25", "vanilla_bge", "ppr_entity", "ppr_chunk", "ppr_entity_loo", "ppr_chunk_loo"]
    labels = {"bm25": "BM25", "vanilla_bge": "BGE (dense)", "ppr_entity": "BGE + entity-graph PPR",
              "ppr_chunk": "BGE + chunk-graph PPR", "ppr_entity_loo": "BGE + entity-graph LOO-PPR*",
              "ppr_chunk_loo": "BGE + chunk-graph LOO-PPR*"}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS) + 1.6, 3.9), sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        cis = method_cis(all_rows[ds], ds, configs)
        ys = np.arange(len(configs))[::-1]
        for y, cfg in zip(ys, configs):
            mean, lo, hi = cis[cfg]
            key = "entity" if "entity" in cfg else "chunk" if "chunk" in cfg else cfg
            ax.barh(y, mean, height=0.62, color=COLOR[key], hatch="////" if cfg.endswith("_loo") else None,
                    edgecolor="white", linewidth=0)
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3)
            ax.text(hi + 0.012, y, f"{mean:.3f}", va="center", fontsize=8.5, color=TEXT)
        bge = cis["vanilla_bge"][0]
        ax.axvline(bge, color=COLOR["vanilla_bge"], linestyle=(0, (3, 3)), linewidth=1)
        ax.set_yticks(ys, [labels[c] for c in configs])
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
        ax.set_xlabel("Complete support@5", fontsize=9, color=MUTED)
        style_axes(ax)
    fig.text(0.01, 0.005, "Bars: mean over questions; whiskers: 95% question-bootstrap CI (10,000 replicates). "
             "Dashed line: BGE. Complete support@5 = all gold passages in the top 5.\n"
             "*Hatched = leave-one-out (LOO) variant: exploratory on HotpotQA and the first MuSiQue sample; "
             "its MuSiQue gain was then tested, as pre-registered, on the disjoint replication sample.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_gold_movement(gold_rows, path):
    """Gold passages entering vs leaving the top 5 relative to BGE (what moves complete support@5)."""
    configs = [("ppr_entity", "Entity\nPPR"), ("ppr_chunk", "Chunk\nPPR"),
               ("ppr_entity_loo", "Entity\nLOO-PPR*"), ("ppr_chunk_loo", "Chunk\nLOO-PPR*")]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 3.7), sharey=True)
    width = 0.38
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        rec = {r["config"]: r for r in gold_rows if r["dataset"] == ds}
        xs = np.arange(len(configs))
        for i, (c, _) in enumerate(configs):
            key = "entity" if "entity" in c else "chunk"
            hatch = "////" if c.endswith("_loo") else None
            entered, left = rec[c]["entered_top5"], rec[c]["left_top5"]
            ax.bar(xs[i] - width / 2 - 0.01, entered, width, color=COLOR[key], hatch=hatch, edgecolor="white", linewidth=0)
            ax.bar(xs[i] + width / 2 + 0.01, -left, width, color="#c3c2b7", hatch=hatch, edgecolor="white", linewidth=0)
            ax.text(xs[i] - width / 2, entered + 4, (f"+{entered}" if entered else "0"), ha="center", fontsize=8, color=TEXT)
            ax.text(xs[i] + width / 2, -left - 4, (f"\u2212{left}" if left else "0"), ha="center", va="top", fontsize=8, color=TEXT)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(xs, [lbl for _, lbl in configs], fontsize=8, rotation=0)
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Gold passages", fontsize=9, color=MUTED)
    ymax = max(abs(v) for a in axes for v in a.get_ylim())
    for a in axes:
        a.set_ylim(-ymax * 1.12, ymax * 1.12)
    fig.text(0.01, 0.005, "Above zero (colored): gold passages outside BGE's top 5 that the method moved into the top 5. "
             "Below zero (gray): gold passages in BGE's top 5 that the method pushed out. *EXPLORATORY variant.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_lambda(lam_rows, path):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 3.4), sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        for g, label in (("entity", "Entity PPR"), ("chunk", "Chunk PPR")):
            pts = sorted((r["lambda"], r) for r in lam_rows if r["dataset"] == ds and r["graph"] == g and r["metric"] == "complete_support_at_5")
            x = [p[0] for p in pts]
            y = [p[1]["mean_paired_difference"] for p in pts]
            lo = [p[1]["mean_paired_difference"] - p[1]["diff_ci95_low"] for p in pts]
            hi = [p[1]["diff_ci95_high"] - p[1]["mean_paired_difference"] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], color=COLOR[g], marker="o", markersize=6, linewidth=2, capsize=3, label=label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0.5, color=GRID, linewidth=6, zorder=0)
        ax.set_xticks([0.25, 0.5, 0.75], ["0.25*", "0.5 (pre-reg.)", "0.75*"])
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
        ax.set_xlabel("λ (graph weight)", fontsize=9, color=MUTED)
        ax.set_ylabel("Δ complete support@5 vs BGE", fontsize=9, color=MUTED)
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, "Paired difference with 95% bootstrap CI. *λ = 0.25 and 0.75 are EXPLORATORY.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/graph_ranking_v1")
    parser.add_argument("--analysis-dir", default="results/graph_ranking_v1/analysis_final")
    parser.add_argument("--figures-dir", default="results/graph_ranking_v1/figures/final")
    args = parser.parse_args()
    root = ROOT / args.results_root
    out = ROOT / args.analysis_dir
    figs = ROOT / args.figures_dir
    for d in (out, figs):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {d}")
        d.mkdir(parents=True, exist_ok=True)

    all_rows = {ds: load(root / ds) for ds in DATASETS}
    tables = {k: [] for k in ("headline", "by_group", "loo", "lambda", "top5", "gold_move", "conditional", "headroom")}
    raws = {}
    for ds, rows in all_rows.items():
        with (root / ds / "paired_headline_contrasts.csv").open() as h:
            tables["headline"] += [{"dataset": ds, **r} for r in csv.DictReader(h)]
        tables["by_group"] += by_group_contrasts(rows, ds)
        tables["loo"] += loo_contrasts(rows, ds)
        tables["lambda"] += lambda_sensitivity(rows, ds)
        tables["top5"] += top5_changes(rows, ds)
        s, raws[ds] = gold_rank_movement(rows, ds)
        tables["gold_move"] += s
        tables["conditional"] += conditional_on_bge(rows, ds)
        tables["headroom"] += headroom(rows, ds)
    names = {"headline": "preregistered_headline_contrasts_both.csv", "by_group": "exploratory_by_question_type.csv",
             "loo": "exploratory_loo_contrasts.csv", "lambda": "exploratory_lambda_sensitivity.csv",
             "top5": "exploratory_top5_changes.csv", "gold_move": "exploratory_gold_rank_movement.csv",
             "conditional": "exploratory_conditional_on_bge_failure.csv", "headroom": "exploratory_headroom.csv"}
    for k, rows in tables.items():
        dump_csv(out / names[k], rows)
    group_counts = {ds: dict(Counter(r["question_type"] for r in rows if r["config"] == "vanilla_bge")) for ds, rows in all_rows.items()}
    (out / "analysis_manifest.json").write_text(json.dumps({"inputs": {ds: str((root / ds / "question_results.jsonl").relative_to(ROOT)) for ds in DATASETS},
                                                            "question_group_counts": group_counts, "outputs": names,
                                                            "note": "All tables except the headline file are EXPLORATORY."}, indent=2) + "\n")
    fig_complete_support(all_rows, figs / "fig1_complete_support_at5.png")
    fig_gold_movement(tables["gold_move"], figs / "fig2_gold_top5_entry_exit.png")
    fig_lambda(tables["lambda"], figs / "fig3_lambda_sensitivity.png")
    print(f"wrote {len(names)} tables to {out} and 3 figures to {figs}")


if __name__ == "__main__":
    main()
