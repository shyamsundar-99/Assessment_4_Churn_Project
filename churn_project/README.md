# FreshBasket loyalty churn prediction

Predicts which loyalty members will make **no purchase in the next 3 months**, tested on the official Apr-Jun 2024 outcome.

## Run in VS Code
1. Extract `Assessment_4_Churn_Project.zip`.
2. In VS Code, choose **File → Open Folder** and select the extracted `churn_project` folder (the folder containing `run_all.py`).
3. Install the recommended Python and Jupyter extensions when prompted.
4. Open **Terminal → Run Task → Setup Python environment** once. This creates `.venv` and installs the packages from `requirements.txt`.
5. Select **Run churn pipeline** from **Terminal → Run Task**, or press **F5** and choose **FreshBasket: run pipeline**.

Use Python 3.10 or newer. The equivalent setup from the VS Code terminal is:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```
The pipeline reads the supplied workbook and regenerates the tables, figures, test predictions and forward scores under `outputs/`. To rebuild and execute the EDA notebook, run:
```bash
python build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/01_data_quality_and_eda.ipynb
```
The dataset is expected at `data/FreshBasket_Loyalty_Churn_Dataset.xlsx`. All seeds are fixed (`src/config.py`).

## Layout
| Path | Purpose |
|---|---|
| `src/data.py` | Load (header on row 2), clean, log every fix, build the customer x month panel |
| `src/features.py` | Leakage-safe snapshot features and labels |
| `src/modeling.py` | Rule baseline, logistic regression, LightGBM and class-weighted LightGBM; rolling-origin CV; thresholds; bootstrap CIs; ablation |
| `src/analysis.py` | SHAP drivers, segment tests, gains, targeting economics, what-if scenarios |
| `src/plots.py` | Figures |
| `notebooks/01_data_quality_and_eda.ipynb` | Executed data-quality and EDA notebook |
| `outputs/tables/` | Every result table (metrics, CV folds, ablation, segments, economics, DQ log, robustness) |
| `outputs/figures/` | Figures used in the report and deck |
| `outputs/predictions_test_snapshot_Mar2024.csv` | Probability, predicted label, risk band, top-3 drivers, actual outcome per member |
| `outputs/scores_forward_Jul-Sep2024.csv` | Same scoring for next quarter (no outcome yet) |

## Design in one paragraph
Only one label window is official, so quarterly **snapshots** (Jun, Sep, Dec 2023 for training; Mar 2024 for test; Jun 2024 for forward scoring) apply the same definition (no purchase in the following 3 months, among members who had bought before). Features use only months up to the snapshot. About 80% of churners had already lapsed at the test snapshot, so every metric is reported for **all eligible members** and for **still-active members** (the realistic task), against a recency-rule baseline.

## Key assumptions to confirm
- Missing member-months are treated as zero activity (rows stop appearing as members lapse).
- Retention economics (25% margin, $3 per contact, 4 quarters of value, 20% uplift) are illustrative placeholders, not estimates.
- What-if scenarios are associational and should not be read as causal effects.

## Output interpretation
- Use `population=active` in `outputs/tables/test_metrics.csv` to assess early-warning performance. The all-member score is much easier because it includes members already lapsed at the score date.
- The forward July-September 2024 file has probabilities but no observed outcome in the supplied dataset.
- The supplied scenario and retention-economics outputs use illustrative assumptions. Replace them with approved business inputs and measure uplift in a randomized test.
