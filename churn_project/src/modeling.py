"""Models, time-ordered validation, metrics and ablation."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             brier_score_loss, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config as C
from . import features as F

ALL_FEATURES = [c for g in F.GROUPS.values() for c in g]
NUM = [c for c in ALL_FEATURES if c not in F.CATEGORICAL]


def matrix(df, cols):
    X = df[cols].copy()
    for c in F.CATEGORICAL:
        if c in X:
            X[c] = pd.Categorical(X[c], categories=F.ALL_CATS[c])
    return X


# ----------------------------------------------------------------- models
def make_lgbm(balanced=False, **kw):
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=C.SEED, verbose=-1,
                  class_weight="balanced" if balanced else None)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def make_logreg(cols):
    cat = [c for c in cols if c in F.CATEGORICAL]
    num = [c for c in cols if c not in F.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", categories=[F.ALL_CATS[c] for c in cat]), cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))])


class RuleModel:
    """Baseline: a member is 'churned' if there was no purchase in the last 3 months."""
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = np.clip(X["recency_months"].fillna(12).values, 0, 12) / 12.0
        return np.c_[1 - s, s]
    @staticmethod
    def label(X): return (X["tx_l3"].values == 0).astype(int)


def fit_predict(name, train, test, cols=ALL_FEATURES):
    Xtr, Xte, ytr = matrix(train, cols), matrix(test, cols), train.churn.astype(int).values
    if name == "rule":
        return RuleModel().predict_proba(test)[:, 1], None
    if name == "logreg":
        m = make_logreg(cols).fit(Xtr, ytr)
    elif name == "lgbm":
        m = make_lgbm().fit(Xtr, ytr)
    elif name == "lgbm_balanced":
        m = make_lgbm(balanced=True).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


# ----------------------------------------------------------------- metrics
def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(th[np.nanargmax(f1)])


def metrics(y, p, thr, top_frac=0.10):
    y = np.asarray(y).astype(int)
    out = {"n": len(y), "base_rate": y.mean()}
    if y.min() == y.max():
        return out
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    k = max(1, int(round(top_frac * len(y))))
    top = np.argsort(-p)[:k]
    out.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), precision=pr, recall=rc, f1=f1,
               accuracy=(pred == y).mean(), brier=brier_score_loss(y, np.clip(p, 0, 1)),
               precision_at_top10=y[top].mean(), lift_at_top10=y[top].mean() / y.mean(), threshold=thr)
    return out


def bootstrap_ci(y, p, n=300, seed=C.SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); res = {"roc_auc": [], "pr_auc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        res["roc_auc"].append(roc_auc_score(y[i], p[i])); res["pr_auc"].append(average_precision_score(y[i], p[i]))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in res.items()}


# ----------------------------------------------------------------- validation
def eligible(df):
    return df[df.eligible == 1]


def rolling_cv(snaps, model_names, cols=ALL_FEATURES):
    """Rolling-origin CV: [Jun -> Sep], [Jun+Sep -> Dec]. Returns metric rows and fold-2 predictions."""
    o = C.TRAIN_ORIGINS
    folds = [([o[0]], o[1]), ([o[0], o[1]], o[2])]
    rows, preds = [], {}
    for tr_o, va_o in folds:
        tr = eligible(pd.concat([snaps[x] for x in tr_o])); va = eligible(snaps[va_o])
        for m in model_names:
            p, _ = fit_predict(m, tr, va, cols)
            for pop, mask in [("all", np.ones(len(va), bool)), ("active", va.active_at_snapshot.values == 1)]:
                r = metrics(va.churn.values[mask], p[mask], 0.5)
                r.update(model=m, population=pop, fold=f"train {'+'.join(x[:7] for x in tr_o)} -> val {va_o[:7]}")
                rows.append(r)
            if va_o == o[2]:
                preds[m] = (va, p)
    return pd.DataFrame(rows), preds


def tune_thresholds(preds):
    """F1-optimal threshold per model, tuned on the Dec-2023 validation fold (earlier than test)."""
    th = {}
    for m, (va, p) in preds.items():
        act = va.active_at_snapshot.values == 1
        th[m] = {"all": best_f1_threshold(va.churn.values, p) if m != "rule" else 0.25,
                 "active": best_f1_threshold(va.churn.values[act], p[act]) if m != "rule" else 1.0}
    return th


def final_test(snaps, model_names, thresholds, cols=ALL_FEATURES):
    tr = eligible(pd.concat([snaps[o] for o in C.TRAIN_ORIGINS])); te = eligible(snaps[C.TEST_ORIGIN])
    rows, preds, models = [], {}, {}
    for m in model_names:
        p, mod = fit_predict(m, tr, te, cols)
        preds[m] = p; models[m] = mod
        for pop, mask in [("all", np.ones(len(te), bool)), ("active", te.active_at_snapshot.values == 1)]:
            thr = thresholds[m][pop]
            r = metrics(te.churn.values[mask], p[mask], thr)
            ci = bootstrap_ci(te.churn.values[mask], p[mask])
            r.update(model=m, population=pop, roc_ci=f"{ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}",
                     pr_ci=f"{ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}")
            rows.append(r)
    return pd.DataFrame(rows), preds, models, tr, te


# ----------------------------------------------------------------- ablation
def ablation(snaps, thresholds_thr=None):
    G = F.GROUPS
    steps = [("demographic", G["demographic"]),
             ("+ purchase behaviour", G["demographic"] + G["purchase"]),
             ("+ engagement", G["demographic"] + G["purchase"] + G["engagement"]),
             ("+ support", ALL_FEATURES)]
    drops = [(f"all minus {g}", [c for c in ALL_FEATURES if c not in G[g]]) for g in G]
    drops.append(("all minus recency_months", [c for c in ALL_FEATURES if c != "recency_months"]))
    tr = eligible(pd.concat([snaps[o] for o in C.TRAIN_ORIGINS])); te = eligible(snaps[C.TEST_ORIGIN])
    rows = []
    for kind, lst in [("stepwise", steps), ("drop-one", drops)]:
        for name, cols in lst:
            p, _ = fit_predict("lgbm", tr, te, cols)
            for pop, mask in [("all", np.ones(len(te), bool)), ("active", te.active_at_snapshot.values == 1)]:
                r = metrics(te.churn.values[mask], p[mask], 0.5)
                rows.append(dict(design=kind, feature_set=name, population=pop, n_features=len(cols),
                                 roc_auc=r["roc_auc"], pr_auc=r["pr_auc"], lift_at_top10=r["lift_at_top10"]))
    return pd.DataFrame(rows)
