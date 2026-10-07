"""End-to-end pipeline: python run_all.py  ->  outputs/ (tables, figures, predictions)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src import config as C, data as D, features as F, modeling as M, analysis as A, plots as P


def main():
    prof, act, lab = D.load_raw()
    prof, act, lab, dq = D.clean(prof, act, lab)
    panel = D.build_panel(prof, act)
    dq.to_csv(C.TAB / "data_quality_log.csv", index=False)
    snaps = F.build_all(panel, prof, lab)

    # ------------- snapshot summary
    rows = []
    for o, s in snaps.items():
        e = s[s.eligible == 1]; a = e[e.active_at_snapshot == 1]
        rows.append(dict(snapshot=o[:7], eligible=len(e), active=len(a), churn_rate=e.churn.mean(), active_churn_rate=a.churn.mean(),
                         lapsed_churn_rate=e[e.active_at_snapshot == 0].churn.mean()))
    pd.DataFrame(rows).to_csv(C.TAB / "snapshot_summary.csv", index=False)
    t = snaps[C.TEST_ORIGIN]; t = t[t.eligible == 1]
    pd.DataFrame({"metric": ["derived label == official CHURNED (eligible members)"], "value": [(t.label_derived == t.churn).mean()]}).to_csv(C.TAB / "label_check.csv", index=False)

    # ------------- models
    names = ["rule", "logreg", "lgbm", "lgbm_balanced"]
    cv, cvpreds = M.rolling_cv(snaps, names)
    cv.to_csv(C.TAB / "cv_folds.csv", index=False)
    th = M.tune_thresholds(cvpreds)
    res, preds, models, tr, te = M.final_test(snaps, names, th)
    res.to_csv(C.TAB / "test_metrics.csv", index=False)
    pd.DataFrame(th).T.to_csv(C.TAB / "thresholds.csv")
    pt = preds["lgbm"]; thr = th["lgbm"]["all"]

    # robustness: customer-disjoint split (train on half the customers, test other half, still time-forward)
    ids = np.array(sorted(set(tr.index))); rng = np.random.default_rng(C.SEED); rng.shuffle(ids); half = set(ids[: len(ids) // 2])
    tr_h, te_h = tr[tr.index.isin(half)], te[~te.index.isin(half)]
    ph, _ = M.fit_predict("lgbm", tr_h, te_h); rb = []
    for pop, m in [("all", np.ones(len(te_h), bool)), ("active", te_h.active_at_snapshot.values == 1)]:
        r = M.metrics(te_h.churn.values[m], ph[m], th["lgbm"][pop]); r.update(check="customer-disjoint + time-forward", population=pop); rb.append(r)
    # shuffled-label sanity check (10 seeds)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for sd in range(10):
        trs = tr.copy(); trs["churn"] = np.random.default_rng(sd).permutation(trs.churn.values)
        pp, _ = M.fit_predict("lgbm", trs, te); aucs.append(roc_auc_score(te.churn, pp))
    rb.append(dict(check="shuffled labels, 10 seeds (expect ~0.5)", population="all", roc_auc=float(np.mean(aucs)), n=len(te)))
    pd.DataFrame(rb).to_csv(C.TAB / "robustness_checks.csv", index=False)

    ab = M.ablation(snaps); ab.to_csv(C.TAB / "ablation.csv", index=False)

    # ------------- explainability + predictions (test snapshot)
    Xte = M.matrix(te, M.ALL_FEATURES)
    sv = A.shap_values(models["lgbm"], Xte)
    pd.DataFrame({"feature": [A.NICE[c] for c in Xte.columns], "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(C.TAB / "shap_importance.csv", index=False)
    band = lambda p_: np.where(p_ >= thr, "High", np.where(p_ >= 0.10, "Medium", "Low"))
    out = pd.DataFrame({"customer_id": te.index, "snapshot": C.TEST_ORIGIN[:7], "active_at_snapshot": te.active_at_snapshot.values,
                        "churn_probability": pt.round(4), "predicted_churn": (pt >= thr).astype(int), "risk_band": band(pt),
                        "key_drivers": A.top_drivers(sv, Xte), "actual_churned": te.churn.astype(int).values})
    out.sort_values("churn_probability", ascending=False).to_csv(C.OUT / "predictions_test_snapshot_Mar2024.csv", index=False)

    # ------------- segments, gains, economics
    seg, tests = A.all_segments(te, pt, thr); seg.to_csv(C.TAB / "segment_comparison.csv", index=False); tests.to_csv(C.TAB / "segment_tests.csv", index=False)
    act_m = te.active_at_snapshot.values == 1
    A.gains_table(te.churn.values[act_m], pt[act_m]).to_csv(C.TAB / "gains_active.csv", index=False)
    A.gains_table(te.churn.values, pt).to_csv(C.TAB / "gains_all.csv", index=False)
    econ, rnd, qspend = A.targeting_economics(te, pt); econ.to_csv(C.TAB / "targeting_economics.csv", index=False); rnd.to_csv(C.TAB / "targeting_random.csv", index=False)

    # ------------- production model + forward scoring (Jul-Sep 2024)
    trall = M.eligible(pd.concat([snaps[o] for o in C.TRAIN_ORIGINS + [C.TEST_ORIGIN]]))
    prod = M.make_lgbm().fit(M.matrix(trall, M.ALL_FEATURES), trall.churn.astype(int))
    fw = M.eligible(snaps[C.FORWARD_ORIGIN]); Xf = M.matrix(fw, M.ALL_FEATURES); pf = prod.predict_proba(Xf)[:, 1]
    svf = A.shap_values(prod, Xf)
    pd.DataFrame({"customer_id": fw.index, "snapshot": C.FORWARD_ORIGIN[:7], "active_at_snapshot": fw.active_at_snapshot.values,
                  "churn_probability": pf.round(4), "predicted_churn": (pf >= thr).astype(int), "risk_band": band(pf),
                  "key_drivers": A.top_drivers(svf, Xf)}).sort_values("churn_probability", ascending=False).to_csv(C.OUT / "scores_forward_Jul-Sep2024.csv", index=False)
    wi = A.scenario_whatif(prod, snaps[C.FORWARD_ORIGIN]); wi.to_csv(C.TAB / "scenario_whatif.csv", index=False)

    # ------------- figures
    P.lapse_timeline(lab); P.missing_rows(panel, prof); P.pre_churn_decay(act, lab)
    P.roc_pr(te, preds, ["rule", "logreg", "lgbm"], th); P.gains(te, pt); P.calibration(te, pt)
    P.shap_plots(sv, Xte); P.ablation(ab); P.segments(seg); P.economics(econ)
    print("done. Test metrics:\n", res[["model", "population", "roc_auc", "pr_auc", "precision", "recall", "f1"]].round(3).to_string())
    print("avg quarterly spend (active):", round(qspend, 1))


if __name__ == "__main__":
    main()
