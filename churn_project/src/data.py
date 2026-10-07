"""Loading and cleaning of the three FreshBasket loyalty tables, with a data-quality log."""
import numpy as np
import pandas as pd
from . import config as C


def load_raw(path=C.DATA_PATH):
    """The workbook has a title banner in row 1, so the real header is on row 2 (header=1)."""
    x = pd.ExcelFile(path)
    prof = x.parse("fb Customer Profile", header=1)
    act = x.parse("fb Monthly Activity", header=1)
    lab = x.parse("fb Churn Label", header=1)
    return prof, act, lab


def clean(prof, act, lab):
    """Return cleaned (profile, activity, label) and a DataFrame logging every fix."""
    log = []

    def note(table, issue, n, action):
        log.append({"table": table, "issue": issue, "rows_affected": int(n), "action": action})

    prof, act, lab = prof.copy(), act.copy(), lab.copy()

    # ---------------- profile ----------------
    n = prof.duplicated().sum(); note("profile", "exact duplicate rows", n, "dropped")
    prof = prof.drop_duplicates()
    note("profile", "duplicate CUSTOMER_ID after exact-dup removal", prof.CUSTOMER_ID.duplicated().sum(), "none left")
    for col in ["CITY", "MEMBERSHIP_TIER"]:
        raw = prof[col].astype(str)
        clean_ = raw.str.strip().str.title()
        note("profile", f"inconsistent casing/whitespace in {col}", (raw != clean_).sum(), "strip + title-case")
        prof[col] = clean_
    note("profile", "missing AGE", prof.AGE.isna().sum(), "median impute + missing flag (in features)")
    note("profile", "missing GENDER", prof.GENDER.isna().sum(), "label 'Unknown'")
    note("profile", "missing HOME_STORE_PRICE_TIER", prof.HOME_STORE_PRICE_TIER.isna().sum(), "label 'Unknown'")
    prof["GENDER"] = prof.GENDER.fillna("Unknown")
    prof["HOME_STORE_PRICE_TIER"] = prof.HOME_STORE_PRICE_TIER.fillna("Unknown")
    prof["SIGNUP_DATE"] = pd.to_datetime(prof.SIGNUP_DATE)

    # ---------------- label ----------------
    n = lab.duplicated().sum(); note("label", "exact duplicate rows", n, "dropped")
    lab = lab.drop_duplicates()
    note("label", "missing LAST_PURCHASE_DATE", lab.LAST_PURCHASE_DATE.isna().sum(), "kept (no purchase in window)")

    # ---------------- activity ----------------
    act["MONTH"] = pd.to_datetime(act.MONTH)
    n = act.duplicated().sum(); note("activity", "exact duplicate rows", n, "dropped")
    act = act.drop_duplicates()
    key = ["CUSTOMER_ID", "MONTH"]
    n = act.duplicated(key).sum()
    note("activity", "conflicting duplicates for same customer-month (sign-flipped spend)", n, "kept the non-negative row")
    act = act.sort_values(key + ["TOTAL_SPEND"], ascending=[True, True, False]).drop_duplicates(key)

    neg = (act.TOTAL_SPEND < 0)
    note("activity", "negative TOTAL_SPEND", neg.sum(), "abs() if TRANSACTIONS>0 else 0")
    act.loc[neg, "TOTAL_SPEND"] = np.where(act.loc[neg, "TRANSACTIONS"] > 0, act.loc[neg, "TOTAL_SPEND"].abs(), 0.0)

    z = (act.TRANSACTIONS == 0) & (act.TOTAL_SPEND > 0)
    note("activity", "spend>0 but TRANSACTIONS==0", z.sum(), "spend set to 0 (no purchase recorded)")
    act.loc[z, "TOTAL_SPEND"] = 0.0
    z2 = (act.TRANSACTIONS > 0) & (act.TOTAL_SPEND == 0)
    note("activity", "TRANSACTIONS>0 but spend==0", z2.sum(), "kept (could be fully-discounted baskets); flagged only")

    nz = act.loc[act.TOTAL_SPEND > 0, "TOTAL_SPEND"]
    cap = nz.quantile(C.OUTLIER_QUANTILE)
    out = act.TOTAL_SPEND > cap
    note("activity", f"outlier TOTAL_SPEND > {C.OUTLIER_QUANTILE:.1%} quantile (${cap:,.0f})", out.sum(), "winsorised at cap")
    act["TOTAL_SPEND"] = act.TOTAL_SPEND.clip(upper=cap)

    ids = set(prof.CUSTOMER_ID) & set(lab.CUSTOMER_ID)
    note("all", "customers not present in all of profile/label", len(set(prof.CUSTOMER_ID) ^ set(lab.CUSTOMER_ID)), "dropped from modelling")
    prof, lab = prof[prof.CUSTOMER_ID.isin(ids)], lab[lab.CUSTOMER_ID.isin(ids)]
    act = act[act.CUSTOMER_ID.isin(ids)]

    months = pd.date_range(C.DATA_START, C.DATA_END, freq="MS")
    exp = []
    for cid, su in prof.set_index("CUSTOMER_ID").SIGNUP_DATE.items():
        start = max(su.to_period("M").to_timestamp(), pd.Timestamp(C.DATA_START))
        exp.append(pd.DataFrame({"CUSTOMER_ID": cid, "MONTH": months[months >= start]}))
    grid = pd.concat(exp)
    miss = grid.merge(act[key], on=key, how="left", indicator=True)
    note("activity", "missing customer-months (no row between signup and Jun-2024)", (miss._merge == "left_only").sum(),
         "treated as zero activity: rows stop appearing once members lapse (affected customers grow monotonically from about 140 in Jan-23 to about 1,050 in Jun-24)")
    note("activity", "customers with no activity rows at all", len(ids - set(act.CUSTOMER_ID)), "kept in profile, never eligible")

    return prof.reset_index(drop=True), act.reset_index(drop=True), lab.reset_index(drop=True), pd.DataFrame(log)


def build_panel(prof, act):
    """Complete customer x month panel from max(signup month, Jan-2023) to Jun-2024.

    A customer-month with no recorded row is treated as *no activity* (all counts = 0).
    `row_present` records whether a real row existed. Ratio fields (avg basket, promo %)
    stay NaN when there were no transactions.
    """
    months = pd.date_range(C.DATA_START, C.DATA_END, freq="MS")
    frames = []
    for cid, su in prof.set_index("CUSTOMER_ID").SIGNUP_DATE.items():
        start = max(su.to_period("M").to_timestamp(), pd.Timestamp(C.DATA_START))
        frames.append(pd.DataFrame({"CUSTOMER_ID": cid, "MONTH": months[months >= start]}))
    grid = pd.concat(frames, ignore_index=True)
    panel = grid.merge(act, on=["CUSTOMER_ID", "MONTH"], how="left")
    panel["row_present"] = panel.TRANSACTIONS.notna().astype(int)
    count_cols = ["TRANSACTIONS", "TOTAL_SPEND", "DISTINCT_CATEGORIES", "APP_SESSIONS", "EMAILS_OPENED",
                  "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]
    panel[count_cols] = panel[count_cols].fillna(0)
    return panel
