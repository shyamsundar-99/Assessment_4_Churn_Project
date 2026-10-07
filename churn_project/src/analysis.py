"""Segment comparison, explainability, targeting economics and retention scenarios."""
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score, average_precision_score
from . import config as C
from . import features as F
from . import modeling as M

NICE = {
    "recency_months": "Months since last purchase", "tx_l1": "Transactions (last month)", "tx_l3": "Transactions (last 3m)",
    "tx_p3": "Transactions (prior 3m)", "tx_trend": "Transaction trend (last3/prior3)", "tx_l6": "Transactions (last 6m)",
    "spend_l3": "Spend (last 3m)", "spend_p3": "Spend (prior 3m)", "spend_trend": "Spend trend", "avg_basket_l3": "Avg basket (last 3m)",
    "months_active_l6": "Active months (last 6)", "zero_months_l3": "Zero-purchase months (last 3)", "tx_cv_l6": "Purchase volatility (6m)",
    "categories_l3": "Categories bought (last 3m)", "missing_rows_l6": "Months with no record (last 6)",
    "app_l3": "App sessions (last 3m)", "app_p3": "App sessions (prior 3m)", "app_trend": "App-session trend", "emails_l3": "Emails opened (last 3m)",
    "emails_trend": "Email-open trend", "coupons_l3": "Coupons redeemed (last 3m)", "coupons_l6": "Coupons redeemed (last 6m)",
    "promo_pct_l3": "Promo share of purchases", "app_per_tx_l3": "App sessions per purchase", "tickets_l3": "Support tickets (last 3m)",
    "tickets_l6": "Support tickets (last 6m)", "tickets_trend": "Support-ticket trend", "complaint_l6": "Complaint (last 6m)",
    "complaint_l3": "Complaint (last 3m)", "age": "Age", "age_missing": "Age missing", "gender": "Gender", "membership_tier": "Membership tier",
    "marketing_opt_in": "Marketing opt-in", "home_price_tier": "Home-store price tier", "preferred_category": "Preferred category",
    "city": "City", "tenure_months": "Tenure (months)",
}


# ------------------------------------------------------------------ SHAP
def shap_values(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def top_drivers(sv, X, k=3):
    cols = list(X.columns); out = []
    for i in range(len(X)):
        idx = np.argsort(-np.abs(sv[i]))[:k]
        parts = []
        for j in idx:
            v = X.iloc[i, j]
            v = f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v)
            parts.append(f"{NICE[cols[j]]} = {v} ({'raises' if sv[i, j] > 0 else 'lowers'} risk)")
        out.append("; ".join(parts))
    return out


# ------------------------------------------------------------------ segments
def segment_table(te, p, by, thr):
    d = te.assign(p=p, pred=(p >= thr).astype(int))
    g = d.groupby(by, observed=True)
    t = g.agg(members=("churn", "size"), churn_rate=("churn", "mean"), avg_predicted_risk=("p", "mean"),
              pct_active_at_snapshot=("active_at_snapshot", "mean")).reset_index().rename(columns={by: "segment"})
    t.insert(0, "dimension", by)
    a = d[d.active_at_snapshot == 1].groupby(by, observed=True).agg(
        active_members=("churn", "size"), active_churn_rate=("churn", "mean")).reset_index().rename(columns={by: "segment"})
    return t.merge(a, on="segment", how="left")


def chi2_p(te, by, active_only=False):
    d = te[te.active_at_snapshot == 1] if active_only else te
    ct = pd.crosstab(d[by], d.churn)
    return chi2_contingency(ct)[1]


def all_segments(te, p, thr):
    te = te.copy(); te["age_band"] = pd.cut(te.age, [0, 29, 44, 59, 120], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
    te["tenure_band"] = pd.cut(te.tenure_months, [-1, 5, 11, 23, 100], labels=["<6m", "6-11m", "12-23m", "24m+"]).astype(str)
    te["optin"] = te.marketing_opt_in.map({1: "Opted in", 0: "Opted out"})
    dims = ["membership_tier", "signup_cohort", "city", "age_band", "tenure_band", "optin", "home_price_tier", "preferred_category"]
    tabs, tests = [], []
    for d in dims:
        tabs.append(segment_table(te, p, d, thr))
        tests.append({"dimension": d, "chi2_p_all": chi2_p(te, d), "chi2_p_active_only": chi2_p(te, d, True)})
    return pd.concat(tabs, ignore_index=True), pd.DataFrame(tests)


# ------------------------------------------------------------------ gains / calibration
def gains_table(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, bins, labels=False) + 1
    g = d.groupby("decile").agg(members=("y", "size"), churners=("y", "sum"), avg_pred=("p", "mean")).reset_index()
    g["actual_rate"] = g.churners / g.members
    g["cum_capture"] = g.churners.cumsum() / g.churners.sum()
    g["lift"] = g.actual_rate / d.y.mean()
    return g


# ------------------------------------------------------------------ targeting economics (back-test on Mar-24 outcomes)
def targeting_economics(te, p, margin=0.25, contact_cost=3.0, quarters_value=4, uplifts=(0.10, 0.20, 0.30),
                        top_fracs=(0.05, 0.10, 0.20, 0.30)):
    """Back-test on the held-out snapshot using ACTUAL outcomes among still-active members.
    Assumptions (illustrative, to be replaced with the business's real numbers):
      margin: gross margin on incremental spend; contact_cost: cost per member contacted;
      quarters_value: quarters of retained spend credited per saved member;
      uplift: relative reduction in churn probability among contacted members (needs an A/B test to confirm)."""
    d = te.assign(p=p); d = d[d.active_at_snapshot == 1].sort_values("p", ascending=False)
    qspend = d.spend_l3.mean()
    rows = []
    for f in top_fracs:
        k = int(round(f * len(d))); c = d.head(k)
        for u in uplifts:
            saved = c.churn.sum() * u
            value = saved * qspend * margin * quarters_value
            cost = k * contact_cost
            rows.append(dict(contact_top_pct=int(f * 100), members_contacted=k,
                             churners_captured=int(c.churn.sum()), pct_of_active_churners_captured=c.churn.sum() / d.churn.sum(),
                             precision=c.churn.mean(), assumed_uplift=u, churners_saved=saved, gross_value=value, cost=cost,
                             net_value=value - cost, roi=(value - cost) / cost))
        # random targeting at same budget (uplift 20%)
    rnd = []
    for f in top_fracs:
        k = int(round(f * len(d)))
        rnd.append(dict(contact_top_pct=int(f * 100), random_churners_captured=k * d.churn.mean()))
    return pd.DataFrame(rows), pd.DataFrame(rnd), qspend


# ------------------------------------------------------------------ what-if scenarios (model-implied, associational)
def scenario_whatif(model, fwd, cols=M.ALL_FEATURES):
    """Re-score active members at the forward snapshot under hypothetical feature changes."""
    d = fwd[(fwd.eligible == 1) & (fwd.active_at_snapshot == 1)].copy()
    base = model.predict_proba(M.matrix(d, cols))[:, 1]
    out = []

    def run(name, mask, fn, note):
        m = mask(d); x = d.copy(); x.loc[m] = fn(x.loc[m])
        new = model.predict_proba(M.matrix(x, cols))[:, 1]
        out.append(dict(scenario=name, members_targeted=int(m.sum()), baseline_expected_churners=base[m.values].sum(),
                        scenario_expected_churners=new[m.values].sum(),
                        churners_avoided=base[m.values].sum() - new[m.values].sum(),
                        relative_reduction=1 - new[m.values].sum() / max(base[m.values].sum(), 1e-9), note=note))

    def resolve(x):
        x["tickets_l6"] = (x.tickets_l6 - x.tickets_l3).clip(lower=0); x["tickets_l3"] = 0; x["tickets_trend"] = 1.0
        x["complaint_l3"] = 0; x["complaint_l6"] = 0; return x

    run("A. Resolve open support issues", lambda x: (x.tickets_l3 >= 1) | (x.complaint_l3 == 1), resolve,
        "members with a ticket/complaint in last 3m: tickets and complaint flags set to zero")

    def engage(x):
        x["app_l3"] = x.app_l3 + 3; x["app_trend"] = (x.app_l3 + 1) / (x.app_p3 + 1)
        x["emails_l3"] = x.emails_l3 + 2; x["coupons_l3"] = x.coupons_l3 + 1; x["coupons_l6"] = x.coupons_l6 + 1
        x["app_per_tx_l3"] = x.app_l3 / (x.tx_l3 + 1); return x

    run("B. Re-engagement push (app + email + coupon)", lambda x: x.tx_trend < 0.7, engage,
        "members whose purchases fell >30% vs prior 3m: +3 app sessions, +2 emails opened, +1 coupon redeemed")

    def lift_tx(x):
        x["tx_l1"] = x.tx_l1 + 1; x["tx_l3"] = x.tx_l3 + 2; x["tx_l6"] = x.tx_l6 + 2
        x["tx_trend"] = (x.tx_l3 + 1) / (x.tx_p3 + 1); x["spend_l3"] = x.spend_l3 + 2 * x.avg_basket_l3.fillna(30); return x

    run("C. Purchase uplift (+2 transactions in 3m)", lambda x: x.tx_trend < 0.7, lift_tx,
        "same at-risk group; what the model would say if offers translated into two extra purchases (upper bound)")
    return pd.DataFrame(out)
