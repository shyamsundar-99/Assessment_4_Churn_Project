import nbformat as nbf
nb = nbf.v4.new_notebook(); c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s)); code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# FreshBasket loyalty churn: data quality and exploratory analysis
**Goal of this notebook:** understand the three tables, log and fix every data-quality problem, and establish *how churn actually behaves* before modelling.
Run from the project root (`jupyter nbconvert --execute notebooks/01_data_quality_and_eda.ipynb`).""")
code("""import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from IPython.display import Image, display
from src import config as C, data as D, features as F
pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.max_colwidth', 110)
prof_raw, act_raw, lab_raw = D.load_raw()
print({'profile': prof_raw.shape, 'activity': act_raw.shape, 'label': lab_raw.shape})
prof_raw.head(3)""")
md("""## 1. Data quality: what was wrong and what we did
The workbook has a title banner above the real header (`header=1`). Issues below match the glossary's warning (duplicates, casing, missing values, negative spend, outliers) plus two found during exploration: *sign-flipped duplicate rows* and *missing customer-months*.""")
code("""prof, act, lab, dq = D.clean(prof_raw, act_raw, lab_raw)
dq""")
md("""**Notes on judgement calls**
- **Conflicting duplicates (2 rows):** the same customer-month appears twice, once with the spend sign flipped. Kept the non-negative row.
- **Negative spend (22 rows):** after de-duplication, remaining negatives are sign errors on months with purchases (`abs`) or zero-purchase months (set to 0).
- **Outliers:** one customer-month showed $3,731 against a 99.9th percentile of about $685; winsorised rather than dropped.
- **Missing customer-months (11,013):** see next section, this one matters for modelling.""")
md("## 2. Missing customer-months are a lapse signal, not random gaps")
code("""panel = D.build_panel(prof, act)
m = panel.groupby('MONTH').row_present.apply(lambda x: (x == 0).sum())
print(m.to_string())
display(Image('../outputs/figures/02_missing_rows.png'))""")
md("""The count of customers with no row grows almost monotonically (about 140 in Jan-23 to about 1,050 by Jun-24). Rows effectively stop appearing as members lapse, so a missing month is **evidence of no activity**, not unknown data. We therefore treat it as zero activity (and keep a `missing_rows_l6` feature). Imputing means or dropping those months would have hidden exactly the behaviour we want to predict.""")
md("## 3. Distributions and outliers (after cleaning)")
code("""fig, ax = plt.subplots(1, 3, figsize=(11, 3))
act_raw.TOTAL_SPEND.clip(upper=800).hist(ax=ax[0], bins=50, color='#9aa0a6'); ax[0].set_title('Raw monthly spend (clipped at 800 for display)')
act.TOTAL_SPEND[act.TOTAL_SPEND > 0].hist(ax=ax[1], bins=50, color='#1f5fa8'); ax[1].set_title('Cleaned monthly spend (>0)')
act.TRANSACTIONS.value_counts().sort_index().plot.bar(ax=ax[2], color='#1f5fa8'); ax[2].set_title('Transactions per month')
plt.tight_layout(); plt.show()
act[['TRANSACTIONS','TOTAL_SPEND','AVG_BASKET_VALUE','APP_SESSIONS','EMAILS_OPENED','COUPONS_REDEEMED','SUPPORT_TICKETS']].describe().round(2)""")
md("## 4. The churn label: structure and a critical finding")
code("""print(lab.CHURNED.value_counts(normalize=True).round(3).to_dict(), '<- overall churn rate (all 2,600 members)')
display(Image('../outputs/figures/01_last_purchase_of_churners.png'))
snaps = F.build_all(panel, prof, lab)
t = snaps[C.TEST_ORIGIN]; e = t[t.eligible == 1]
print('eligible members at 31-Mar-2024 (signed up and had bought before):', len(e))
print(pd.crosstab(e.active_at_snapshot.map({1: 'bought in Jan-Mar 2024', 0: 'no purchase in Jan-Mar 2024'}), e.churn.map({0: 'stayed', 1: 'churned'}), margins=True))
print('label rebuilt from raw activity == official CHURNED:', round((e.label_derived == e.churn).mean(), 4))""")
md("""**Finding:** of 2,368 members who had purchased before April 2024, **653 had already stopped buying before the label window and 649 of them (99.4%) are labelled churned.** Only **164 of the 1,715 still-active members (9.6%)** churn. So ~80% of churn is members who had *already lapsed* and are caught by a one-line recency rule.

Consequences for the analysis: (1) headline AUC on all members would look near-perfect and mean little, so every result is reported **twice**: all eligible members and still-active members; (2) the commercially useful question, *who among today's active customers is about to leave*, is the harder 9.6% problem.""")
md("## 5. Churners fade before they leave")
code("""display(Image('../outputs/figures/03_pre_churn_decay.png'))""")
md("""In the months before their last purchase, churners' purchases and app sessions fall steadily while complaints rise. That is the pre-churn signal the models learn from (and why trend features and engagement matter).""")
md("## 6. Who churns? Raw differences are mostly an exposure artefact")
code("""te = e
for dim in ['membership_tier', 'tenure_band']:
    pass
d = te.assign(tenure_band=pd.cut(te.tenure_months, [-1, 5, 11, 23, 100], labels=['<6m','6-11m','12-23m','24m+']))
print('Churn by tier'); print(d.groupby('membership_tier').churn.agg(['mean','size']).round(3))
print('Churn by tenure band: all members vs still-active only')
print(pd.DataFrame({'all': d.groupby('tenure_band').churn.mean(), 'active_only': d[d.active_at_snapshot==1].groupby('tenure_band').churn.mean()}).round(3))
display(Image('../outputs/figures/10_segments.png'))
pd.read_csv('../outputs/tables/segment_tests.csv').round(4)""")
md("""Raw churn is much lower for new members (<6m: 10.7% vs ~40% for 12m+, p<0.001) **but among still-active members the gap disappears (p=0.61)**. Newer members simply have not had time to lapse. Membership tier, city, age band and preferred category show no meaningful difference either way. Marketing opt-out is higher-risk overall (37.8% vs 32.1%, p=0.005) but only marginal among active members (p=0.06).""")
md("""## 7. Design decisions that follow from the EDA
| Issue | Decision |
|---|---|
| Only one official label window (Apr-Jun 2024) | Build **rolling snapshots** (Jun, Sep, Dec 2023 for training; Mar 2024 for test) using the same definition: no purchase in the next 3 months |
| Leakage risk | Every feature uses only months <= the snapshot; checked by wiping all later months and confirming features are unchanged |
| 80% of churn is already-lapsed | Report all-members **and** still-active metrics; compare against a recency rule |
| Same members appear in several snapshots | Add a customer-disjoint robustness check |
| Missing months = lapse | Zero-fill + `missing_rows_l6` feature |
| New signups after Mar-2024 / no purchase history | Not eligible at the snapshot (cannot have churned by definition) |""")
nb.cells = c
nbf.write(nb, "notebooks/01_data_quality_and_eda.ipynb")
