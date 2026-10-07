"""Leakage-safe snapshot features and labels.

A *snapshot* is a (customer, origin month T) pair. Every feature uses only months <= T.
The label is "no purchase in months T+1..T+3" for customers who had purchased on/before T.
For T = Mar-2024 this equals the official CHURNED flag (Apr-Jun 2024).
"""
import numpy as np
import pandas as pd
from . import config as C

METRICS = ["TRANSACTIONS", "TOTAL_SPEND", "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
           "SUPPORT_TICKETS", "COMPLAINT_FLAG", "DISTINCT_CATEGORIES"]

# feature groups used by the ablation study
GROUPS = {
    "demographic": ["age", "age_missing", "gender", "membership_tier", "marketing_opt_in",
                    "home_price_tier", "preferred_category", "city", "tenure_months"],
    "purchase": ["recency_months", "tx_l1", "tx_l3", "tx_p3", "tx_trend", "tx_l6", "spend_l3", "spend_p3",
                 "spend_trend", "avg_basket_l3", "months_active_l6", "zero_months_l3", "tx_cv_l6",
                 "categories_l3", "missing_rows_l6"],
    "engagement": ["app_l3", "app_p3", "app_trend", "emails_l3", "emails_trend", "coupons_l3", "coupons_l6",
                   "promo_pct_l3", "app_per_tx_l3"],
    "support": ["tickets_l3", "tickets_l6", "tickets_trend", "complaint_l6", "complaint_l3"],
}
CATEGORICAL = ["gender", "membership_tier", "home_price_tier", "preferred_category", "city"]


def _pivot(panel, col, months):
    return panel.pivot(index="CUSTOMER_ID", columns="MONTH", values=col).reindex(columns=months)


def build_snapshot(panel, prof, origin, lab=None):
    """Features (+ label) for every customer at snapshot month `origin`."""
    T = pd.Timestamp(origin)
    allm = pd.date_range(C.DATA_START, C.DATA_END, freq="MS")
    hist = [m for m in allm if m <= T]
    fut = [m for m in allm if T < m <= T + pd.DateOffset(months=C.LABEL_MONTHS)]
    ids = prof.CUSTOMER_ID.values
    P = {c: _pivot(panel, c, allm).reindex(ids) for c in METRICS + ["row_present"]}
    inpanel = P["TRANSACTIONS"].notna()          # month exists for this customer (on/after signup)
    P = {k: v.fillna(0) for k, v in P.items()}

    def win(c, a, b):                              # months T-a+1 .. T-b  (a=3,b=0 -> last 3 months)
        cols = hist[len(hist) - a: len(hist) - b] if b else hist[len(hist) - a:]
        return P[c][cols]

    f = pd.DataFrame(index=ids)
    tx = P["TRANSACTIONS"][hist]
    ever = (tx > 0).any(axis=1)
    last_idx = (tx > 0).values[:, ::-1].argmax(axis=1)           # months since last purchase
    f["recency_months"] = np.where(ever, last_idx, np.nan)
    f["tx_l1"] = win("TRANSACTIONS", 1, 0).sum(axis=1)
    f["tx_l3"] = win("TRANSACTIONS", 3, 0).sum(axis=1)
    f["tx_p3"] = win("TRANSACTIONS", 6, 3).sum(axis=1)
    f["tx_l6"] = win("TRANSACTIONS", 6, 0).sum(axis=1)
    f["tx_trend"] = (f.tx_l3 + 1) / (f.tx_p3 + 1)
    f["spend_l3"] = win("TOTAL_SPEND", 3, 0).sum(axis=1)
    f["spend_p3"] = win("TOTAL_SPEND", 6, 3).sum(axis=1)
    f["spend_trend"] = (f.spend_l3 + 10) / (f.spend_p3 + 10)
    f["avg_basket_l3"] = (f.spend_l3 / f.tx_l3.replace(0, np.nan))
    f["months_active_l6"] = (win("TRANSACTIONS", 6, 0) > 0).sum(axis=1)
    f["zero_months_l3"] = (win("TRANSACTIONS", 3, 0) == 0).sum(axis=1)
    w6 = win("TRANSACTIONS", 6, 0)
    f["tx_cv_l6"] = w6.std(axis=1) / w6.mean(axis=1).replace(0, np.nan)
    f["categories_l3"] = win("DISTINCT_CATEGORIES", 3, 0).mean(axis=1)
    f["missing_rows_l6"] = 6 - win("row_present", 6, 0).sum(axis=1)

    f["app_l3"] = win("APP_SESSIONS", 3, 0).sum(axis=1)
    f["app_p3"] = win("APP_SESSIONS", 6, 3).sum(axis=1)
    f["app_trend"] = (f.app_l3 + 1) / (f.app_p3 + 1)
    f["emails_l3"] = win("EMAILS_OPENED", 3, 0).sum(axis=1)
    f["emails_trend"] = (f.emails_l3 + 1) / (win("EMAILS_OPENED", 6, 3).sum(axis=1) + 1)
    f["coupons_l3"] = win("COUPONS_REDEEMED", 3, 0).sum(axis=1)
    f["coupons_l6"] = win("COUPONS_REDEEMED", 6, 0).sum(axis=1)
    f["app_per_tx_l3"] = f.app_l3 / (f.tx_l3 + 1)
    # transaction-weighted promo share over the last 3 months
    promo = panel.assign(_p=panel.PROMO_TXN_PCT.fillna(0) * panel.TRANSACTIONS).pivot(
        index="CUSTOMER_ID", columns="MONTH", values="_p").reindex(columns=allm).reindex(ids).fillna(0)
    pl3 = promo[hist[-3:]].sum(axis=1)
    f["promo_pct_l3"] = pl3 / f.tx_l3.replace(0, np.nan)

    f["tickets_l3"] = win("SUPPORT_TICKETS", 3, 0).sum(axis=1)
    f["tickets_l6"] = win("SUPPORT_TICKETS", 6, 0).sum(axis=1)
    f["tickets_trend"] = (f.tickets_l3 + 0.5) / (win("SUPPORT_TICKETS", 6, 3).sum(axis=1) + 0.5)
    f["complaint_l3"] = win("COMPLAINT_FLAG", 3, 0).max(axis=1)
    f["complaint_l6"] = win("COMPLAINT_FLAG", 6, 0).max(axis=1)

    pr = prof.set_index("CUSTOMER_ID").reindex(ids)
    f["tenure_months"] = (T.year - pr.SIGNUP_DATE.dt.year) * 12 + (T.month - pr.SIGNUP_DATE.dt.month)
    f["age"] = pr.AGE
    f["age_missing"] = pr.AGE.isna().astype(int)
    f["gender"] = pr.GENDER
    f["membership_tier"] = pr.MEMBERSHIP_TIER
    f["marketing_opt_in"] = pr.MARKETING_OPT_IN
    f["home_price_tier"] = pr.HOME_STORE_PRICE_TIER
    f["preferred_category"] = pr.PREFERRED_CATEGORY
    f["city"] = pr.CITY
    f["signup_cohort"] = pr.SIGNUP_DATE.dt.to_period("Q").astype(str)      # for reporting only, not a model feature

    # eligibility: signed up on/before T and has at least one purchase on/before T
    f["eligible"] = (ever & (f.tenure_months >= 0)).astype(int)
    f["active_at_snapshot"] = (f.tx_l3 > 0).astype(int)
    f["origin"] = T

    # label
    fut_tx = P["TRANSACTIONS"][fut].sum(axis=1) if fut else pd.Series(np.nan, index=ids)
    f["label_derived"] = (fut_tx == 0).astype(float) if fut else np.nan
    if lab is not None and T == pd.Timestamp(C.TEST_ORIGIN):
        f["churn"] = lab.set_index("CUSTOMER_ID").CHURNED.reindex(ids).values          # official label
    else:
        f["churn"] = f.label_derived
    f.index.name = "customer_id"
    return f


def build_all(panel, prof, lab):
    origins = C.TRAIN_ORIGINS + [C.TEST_ORIGIN, C.FORWARD_ORIGIN]
    snaps = {o: build_snapshot(panel, prof, o, lab) for o in origins}
    return snaps


def prep(df, cols):
    """Return model matrix with categoricals as pandas category (stable categories)."""
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = pd.Categorical(X[c], categories=sorted(df[c].dropna().unique().tolist()) if False else
                                  ALL_CATS[c])
    return X


ALL_CATS = {
    "gender": ["F", "M", "Other", "Unknown"],
    "membership_tier": ["Silver", "Gold", "Platinum"],
    "home_price_tier": ["Value", "Mainstream", "Upscale", "Unknown"],
    "preferred_category": ["Bakery", "Beverages", "Dairy", "Frozen Foods", "Household", "Personal Care", "Produce", "Snacks"],
    "city": ["Austin", "Boise", "Columbus", "Denver", "Madison", "Nashville", "Portland", "Raleigh", "Spokane", "Tulsa"],
}
