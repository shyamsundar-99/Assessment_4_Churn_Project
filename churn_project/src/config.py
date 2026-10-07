"""Central configuration for the FreshBasket churn project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in (OUT, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
DATA_START = "2023-01-01"
DATA_END = "2024-06-01"          # last month in the data (first-of-month stamps)
LABEL_MONTHS = 3                 # churn = zero purchases in the 3 months after the snapshot

# Snapshot ("origin") months: features use activity up to and including this month,
# label uses the following 3 months. Training origins are chosen so that their label
# windows have fully matured before the test snapshot is created.
TRAIN_ORIGINS = ["2023-06-01", "2023-09-01", "2023-12-01"]
TEST_ORIGIN = "2024-03-01"       # label window Apr-Jun 2024 == the official CHURNED label
FORWARD_ORIGIN = "2024-06-01"    # production scoring: next quarter (Jul-Sep 2024), no label yet

OUTLIER_QUANTILE = 0.999         # winsorise monthly spend at this quantile
