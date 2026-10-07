"""All figures. Consistent, minimal style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_curve, precision_recall_curve
from . import config as C
from . import analysis as A

BLUE, ORANGE, GREY, RED, GREEN = "#1f5fa8", "#e8833a", "#9aa0a6", "#c0392b", "#2e8b57"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(name):
    plt.savefig(C.FIG / name); plt.close()


def lapse_timeline(lab):
    ch = lab[(lab.CHURNED == 1) & lab.LAST_PURCHASE_DATE.notna()]
    s = ch.LAST_PURCHASE_DATE.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = [BLUE if p <= pd.Period("2024-03") else RED for p in s.index]
    ax.bar([p.to_timestamp() for p in s.index], s.values, width=25, color=cols)
    ax.axvline(pd.Timestamp("2024-03-31"), color="k", ls="--", lw=1)
    ax.set_title("Most churners stopped buying long before the label window"); ax.set_ylabel("Churners (by last purchase month)")
    ax.text(pd.Timestamp("2024-03-20"), s.max() * 0.95, "snapshot\n31 Mar 2024", ha="right", va="top", fontsize=8)
    _save("01_last_purchase_of_churners.png")


def missing_rows(panel, prof):
    p = panel[panel.MONTH >= "2023-01-01"]
    s = p.groupby("MONTH").row_present.apply(lambda x: (x == 0).sum())
    fig, ax = plt.subplots(figsize=(8, 3.3)); ax.plot(s.index, s.values, color=ORANGE, marker="o")
    ax.set_title("Customers with no activity row, by month"); ax.set_ylabel("Customers"); _save("02_missing_rows.png")


def pre_churn_decay(act, lab):
    ch = lab[(lab.CHURNED == 1) & lab.LAST_PURCHASE_DATE.notna()].set_index("CUSTOMER_ID").LAST_PURCHASE_DATE
    d = act.join(ch.rename("lp"), on="CUSTOMER_ID", how="inner")
    d["k"] = (d.MONTH.dt.year - d.lp.dt.year) * 12 + d.MONTH.dt.month - d.lp.dt.month
    keep = ["TRANSACTIONS", "APP_SESSIONS", "COMPLAINT_FLAG", "EMAILS_OPENED"]
    g = d[(d.k <= 0) & (d.k >= -6)].groupby("k")[keep].mean()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a_, c, t in zip(ax, ["TRANSACTIONS", "APP_SESSIONS", "COMPLAINT_FLAG"], ["Purchases / month", "App sessions / month", "Complaint rate"]):
        a_.plot(g.index, g[c], marker="o", color=BLUE); a_.set_title(t); a_.set_xlabel("Months before last purchase")
    fig.suptitle("Churners fade before they leave: the signal the model uses", fontweight="bold", y=1.05)
    _save("03_pre_churn_decay.png")


def roc_pr(te, preds, names, thr):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for r, (pop, mask) in enumerate([("All eligible members", np.ones(len(te), bool)), ("Still-active members", te.active_at_snapshot.values == 1)]):
        y = te.churn.values[mask]
        for n, col in zip(names, [GREY, ORANGE, BLUE]):
            p = preds[n][mask]
            fpr, tpr, _ = roc_curve(y, p); axes[r, 0].plot(fpr, tpr, color=col, label=n)
            pr, rc, _ = precision_recall_curve(y, p); axes[r, 1].plot(rc, pr, color=col, label=n)
        axes[r, 0].plot([0, 1], [0, 1], "k:", lw=1); axes[r, 0].set_title(f"ROC: {pop}"); axes[r, 0].set_xlabel("FPR"); axes[r, 0].set_ylabel("TPR")
        axes[r, 1].axhline(y.mean(), color="k", ls=":", lw=1); axes[r, 1].set_title(f"Precision-recall: {pop}"); axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
        axes[r, 0].legend(loc="lower right", fontsize=8)
    plt.tight_layout(); _save("04_roc_pr.png")


def gains(te, p):
    m = te.active_at_snapshot.values == 1
    g = A.gains_table(te.churn.values[m], p[m])
    fig, ax = plt.subplots(figsize=(6, 4)); x = np.r_[0, g.decile * 10]; y = np.r_[0, g.cum_capture * 100]
    ax.plot(x, y, marker="o", color=BLUE, label="Model"); ax.plot([0, 100], [0, 100], "k:", label="Random")
    ax.set_xlabel("% of still-active members contacted (highest risk first)"); ax.set_ylabel("% of churners captured")
    ax.set_title("Targeting the top 10-20% finds most churners"); ax.legend(); _save("05_gains.png")


def calibration(te, p):
    m = te.active_at_snapshot.values == 1
    g = A.gains_table(te.churn.values[m], p[m], bins=10)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.plot([0, 1], [0, 1], "k:")
    ax.plot(g.avg_pred, g.actual_rate, marker="o", color=BLUE); ax.set_xlabel("Mean predicted risk (decile)"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration, still-active members"); _save("06_calibration.png")


def shap_plots(sv, X):
    Xn = X.copy(); Xn.columns = [A.NICE[c] for c in X.columns]
    shap.summary_plot(sv, Xn, plot_type="bar", max_display=12, show=False, color=BLUE)
    plt.title("Mean |SHAP|: what drives predicted churn risk"); _save("07_shap_bar.png")
    shap.summary_plot(sv, Xn, max_display=12, show=False); plt.title("SHAP: direction and size of effects"); _save("08_shap_beeswarm.png")


def ablation(ab):
    s = ab[(ab.design == "stepwise") & (ab.population == "active")]
    d = ab[(ab.design == "drop-one") & (ab.population == "active")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].barh(s.feature_set[::-1], s.pr_auc[::-1], color=BLUE); axes[0].set_title("Adding feature groups (active members)"); axes[0].set_xlabel("PR-AUC")
    axes[1].barh(d.feature_set[::-1], d.pr_auc[::-1], color=ORANGE); axes[1].set_title("Removing one group at a time"); axes[1].set_xlabel("PR-AUC")
    axes[1].axvline(s.pr_auc.iloc[-1], color="k", ls="--", lw=1)
    plt.tight_layout(); _save("09_ablation.png")


def segments(seg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, dim, title in [(axes[0], "tenure_band", "Churn by tenure band"), (axes[1], "membership_tier", "Churn by membership tier")]:
        t = seg[seg.dimension == dim]
        order = ["<6m", "6-11m", "12-23m", "24m+"] if dim == "tenure_band" else ["Silver", "Gold", "Platinum"]
        t = t.set_index("segment").loc[order]; x = np.arange(len(t))
        ax.bar(x - 0.2, t.churn_rate * 100, 0.4, color=GREY, label="All members"); ax.bar(x + 0.2, t.active_churn_rate * 100, 0.4, color=BLUE, label="Still-active only")
        ax.set_xticks(x); ax.set_xticklabels(order); ax.set_title(title); ax.set_ylabel("Churn rate (%)")
    axes[0].legend(); plt.tight_layout(); _save("10_segments.png")


def economics(econ):
    e = econ[econ.assumed_uplift == 0.20]
    fig, ax = plt.subplots(figsize=(6, 3.8)); ax.bar(e.contact_top_pct.astype(str) + "%", e.net_value, color=GREEN)
    for x, v in zip(range(len(e)), e.net_value): ax.text(x, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Share of still-active members contacted"); ax.set_ylabel("Net value ($)")
    ax.set_title("Illustrative net value (20% uplift, see assumptions)"); _save("11_economics.png")


def tx_hist_active(te, p):
    pass
