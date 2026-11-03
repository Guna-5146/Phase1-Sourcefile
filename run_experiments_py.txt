"""
Leakage-free evaluation of the stacking fraud detector for the land registry paper.

Produces every number the revised manuscript needs (Tables 3-7, Figures 3-5):
  - stratified k-fold CV, mean +/- SD, for all baselines and the proposed stack
  - ablation of the stack
  - leakage check without the engineered price features
  - holdout confusion matrix, McNemar test, 5x2cv paired t-test
  - threshold / review-band analysis and per-transaction inference latency
  - permutation feature importance

All preprocessing and feature selection are fit inside each training fold.

Usage:
  pip install scikit-learn xgboost lightgbm pandas scipy
  python run_experiments.py --data land_transactions.csv --target is_fraud --out results
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

SEED = 42
PRICE_FEATURES = ["price_vs_market_diff", "price_ratio"]
PRICE_SOURCE_COLS = ["price_inr", "area_acres", "market_value_per_acre"]


# ---------------------------------------------------------------- data

def load_data(path, target, date_col, drop_cols):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        df["year"], df["month"], df["day_of_week"] = d.dt.year, d.dt.month, d.dt.day_of_week
        df = df.drop(columns=[date_col])
    if all(c in df.columns for c in PRICE_SOURCE_COLS):
        expected = df["area_acres"] * df["market_value_per_acre"]
        df["price_vs_market_diff"] = df["price_inr"] - expected
        df["price_ratio"] = df["price_inr"] / expected.replace(0, np.nan)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df.pop(target).astype(int).values
    return df, y


# ---------------------------------------------------------------- feature selection

class StatFilterSelector(BaseEstimator, TransformerMixin):
    """Chi-squared (categorical) U ANOVA F (numeric) filter, fit on the training fold only.

    Keeps columns with p < alpha, then the k with the largest test statistic rank.
    """

    def __init__(self, alpha=0.05, k=16):
        self.alpha = alpha
        self.k = k

    def fit(self, X, y):
        X = pd.DataFrame(X)
        cat = X.select_dtypes(exclude="number").columns.tolist()
        num = [c for c in X.columns if c not in cat]
        rows = []
        if num:
            Xn = SimpleImputer(strategy="median").fit_transform(X[num])
            f, p = f_classif(Xn, y)
            rows += [(c, "anova", fi, pi) for c, fi, pi in zip(num, f, p)]
        if cat:
            Xc = OrdinalEncoder().fit_transform(X[cat].fillna("missing").astype(str))
            c2, p = chi2(Xc, y)
            rows += [(c, "chi2", ci, pi) for c, ci, pi in zip(cat, c2, p)]
        scores = pd.DataFrame(rows, columns=["feature", "test", "stat", "p"])
        scores["rank"] = scores.groupby("test")["stat"].rank(ascending=False, pct=True)
        kept = scores[scores["p"] < self.alpha].sort_values("rank")
        self.scores_ = scores
        self.selected_ = kept["feature"].head(self.k).tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_]


class Encoder(BaseEstimator, TransformerMixin):
    """Impute; one-hot low-cardinality categoricals, ordinal-encode the rest; scale."""

    def __init__(self, max_onehot=20):
        self.max_onehot = max_onehot

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cat = X.select_dtypes(exclude="number").columns.tolist()
        num = [c for c in X.columns if c not in cat]
        low = [c for c in cat if X[c].nunique() <= self.max_onehot]
        high = [c for c in cat if c not in low]
        self.ct_ = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("low", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low),
            ("high", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                      unknown_value=-1))]), high),
        ])
        self.ct_.fit(X)
        return self

    def transform(self, X):
        return self.ct_.transform(pd.DataFrame(X))


# ---------------------------------------------------------------- models

def xgb(pos_weight):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)


def wrap(model, k):
    return Pipeline([("select", StatFilterSelector(k=k)), ("enc", Encoder()), ("clf", model)])


def stack(pos_weight, base=("gnb", "rf", "xgb"), passthrough=True, meta="xgb"):
    pool = {"gnb": GaussianNB(), "rf": rf(), "xgb": xgb(pos_weight)}
    final = xgb(pos_weight) if meta == "xgb" else LogisticRegression(max_iter=2000)
    return StackingClassifier(estimators=[(b, pool[b]) for b in base], final_estimator=final,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              stack_method="predict_proba", passthrough=passthrough, n_jobs=1)


def model_zoo(pos_weight, k):
    zoo = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Gaussian NB": GaussianNB(),
        "Random Forest": rf(),
        "XGBoost": xgb(pos_weight),
    }
    if LGBMClassifier is not None:
        zoo["LightGBM"] = LGBMClassifier(n_estimators=300, is_unbalance=True, random_state=SEED, verbose=-1)
    zoo["Soft voting (GNB+RF+XGB)"] = VotingClassifier(
        [("gnb", GaussianNB()), ("rf", rf()), ("xgb", xgb(pos_weight))], voting="soft")
    zoo["Proposed stack"] = stack(pos_weight)
    return {n: wrap(m, k) for n, m in zoo.items()}


def ablation_zoo(pos_weight, k):
    zoo = {
        "Full stack (GNB+RF+XGB, passthrough, XGB meta)": stack(pos_weight),
        "Without passthrough": stack(pos_weight, passthrough=False),
        "Without GNB": stack(pos_weight, base=("rf", "xgb")),
        "Without RF": stack(pos_weight, base=("gnb", "xgb")),
        "Without XGB base": stack(pos_weight, base=("gnb", "rf")),
        "LR meta-learner": stack(pos_weight, meta="lr"),
    }
    return {n: wrap(m, k) for n, m in zoo.items()}


# ---------------------------------------------------------------- evaluation

SCORING = {
    "Accuracy": "accuracy", "Balanced acc.": "balanced_accuracy", "Precision": "precision",
    "Recall": "recall", "F1": "f1", "ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
    "MCC": "matthews_corrcoef",
}


def cv_table(models, X, y, folds):
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows = []
    for name, m in models.items():
        t0 = time.time()
        r = cross_validate(m, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"Model": name}
        for k in SCORING:
            v = r[f"test_{k}"] * 100
            row[k] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"
        row["Train time/fold (s)"] = f"{np.mean(r['fit_time']):.1f}"
        rows.append(row)
        print(f"  {name}: F1 {row['F1']}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def holdout_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"Threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": accuracy_score(y, yhat) * 100, "Balanced acc.": balanced_accuracy_score(y, yhat) * 100,
            "Precision": precision_score(y, yhat, zero_division=0) * 100,
            "Recall": recall_score(y, yhat) * 100, "F1": f1_score(y, yhat) * 100,
            "ROC-AUC": roc_auc_score(y, p) * 100, "PR-AUC": average_precision_score(y, p) * 100,
            "MCC": matthews_corrcoef(y, yhat) * 100}


def mcnemar(y, a, b):
    """Exact McNemar test on two prediction vectors."""
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def five_by_two_cv(m1, m2, X, y):
    """Dietterich (1998) 5x2cv paired t-test on F1."""
    diffs = []
    for i in range(5):
        cv = StratifiedKFold(2, shuffle=True, random_state=SEED + i)
        d = []
        for tr, te in cv.split(X, y):
            f = [f1_score(y[te], clone(m).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])) for m in (m1, m2)]
            d.append(f[0] - f[1])
        diffs.append(d)
    diffs = np.array(diffs)
    s2 = ((diffs - diffs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t = diffs[0, 0] / np.sqrt(s2.mean())
    return t, 2 * stats.t.sf(abs(t), df=5)


def latency_ms(model, X, n=1000):
    rows = X.sample(min(n, len(X)), random_state=SEED)
    times = []
    for i in range(len(rows)):
        t0 = time.perf_counter()
        model.predict_proba(rows.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.percentile(times, 95)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="is_fraud")
    ap.add_argument("--date-col", default="transaction_date")
    ap.add_argument("--drop", nargs="*", default=[], help="ID / name columns to drop")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cost-fn", type=float, default=10.0, help="cost of a missed fraud")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="cost of a false alarm")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-5x2", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y = load_data(a.data, a.target, a.date_col, a.drop)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    summary = {"n_records": int(len(y)), "n_features_raw": int(X.shape[1]),
               "n_fraud": int(y.sum()), "fraud_rate_pct": round(100 * y.mean(), 2)}
    print("Dataset:", summary)

    print("Table 3: baselines vs proposed (CV)")
    t3 = cv_table(model_zoo(pos_weight, a.k), X, y, a.folds)
    t3.to_csv(f"{a.out}/table3_cv_comparison.csv", index=False)

    print("Table 4: ablation (CV)")
    t4 = cv_table(ablation_zoo(pos_weight, a.k), X, y, a.folds)
    t4.to_csv(f"{a.out}/table4_ablation.csv", index=False)

    print("Table 5: leakage check without engineered price features (CV)")
    Xn = X.drop(columns=[c for c in PRICE_FEATURES if c in X.columns])
    t5 = pd.concat([
        t3[t3["Model"] == "Proposed stack"].assign(Model="All features"),
        cv_table({"Without engineered price features": wrap(stack(pos_weight), a.k)}, Xn, y, a.folds),
    ])
    t5.to_csv(f"{a.out}/table5_leakage_check.csv", index=False)

    print("Holdout analysis (stratified 80/20)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    prop = wrap(stack(pos_weight), a.k).fit(Xtr, ytr)
    p_prop = prop.predict_proba(Xte)[:, 1]
    selected = prop.named_steps["select"].selected_
    prop.named_steps["select"].scores_.to_csv(f"{a.out}/feature_selection_scores.csv", index=False)

    tests = []
    for name in ["XGBoost", "Random Forest"]:
        base = model_zoo(pos_weight, a.k)[name].fit(Xtr, ytr)
        n01, n10, p = mcnemar(yte, (p_prop >= 0.5).astype(int), base.predict(Xte))
        tests.append({"Comparison": f"Proposed vs {name}", "Test": "McNemar (exact)",
                      "Proposed right / other wrong": n01, "Other right / proposed wrong": n10, "p-value": p})
    if not a.skip_5x2:
        t, p = five_by_two_cv(wrap(stack(pos_weight), a.k), model_zoo(pos_weight, a.k)["XGBoost"], X, y)
        tests.append({"Comparison": "Proposed vs XGBoost", "Test": "5x2cv paired t (F1)", "t": t, "p-value": p})
    pd.DataFrame(tests).to_csv(f"{a.out}/table6_significance.csv", index=False)

    thr = pd.DataFrame([holdout_metrics(yte, p_prop, t) for t in np.round(np.arange(0.1, 0.95, 0.05), 2)])
    thr["Expected cost"] = thr["FN"] * a.cost_fn + thr["FP"] * a.cost_fp
    thr.to_csv(f"{a.out}/fig4_threshold_sweep.csv", index=False)
    best = thr.loc[thr["Expected cost"].idxmin(), "Threshold"]
    band = []
    for lo, hi in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]:
        auto = (p_prop < lo) | (p_prop >= hi)
        band.append({"Review band": f"[{lo}, {hi})", "Sent to manual review (%)": 100 * (~auto).mean(),
                     "Auto-decided error rate (%)": 100 * np.mean((p_prop[auto] >= hi) != yte[auto]),
                     "Fraud missed by auto-accept": int(np.sum((p_prop < lo) & (yte == 1)))})
    pd.DataFrame(band).to_csv(f"{a.out}/table7_review_band.csv", index=False)

    holdout = pd.DataFrame([holdout_metrics(yte, p_prop, 0.5), holdout_metrics(yte, p_prop, best)])
    holdout.to_csv(f"{a.out}/holdout_confusion.csv", index=False)

    pi = permutation_importance(prop, Xte, yte, scoring="average_precision", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    imp = (pd.DataFrame({"feature": Xte.columns, "importance_mean": pi.importances_mean,
                         "importance_sd": pi.importances_std})
           .sort_values("importance_mean", ascending=False))
    imp.to_csv(f"{a.out}/fig3_permutation_importance.csv", index=False)

    med, p95 = latency_ms(prop, Xte)
    summary.update({"selected_features": selected, "cost_optimal_threshold": float(best),
                    "latency_ms_median": round(med, 2), "latency_ms_p95": round(p95, 2)})
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(f"{a.out}/tables.md", "w") as f:
        f.write("# Results for the revised manuscript\n\n")
        f.write(f"```\n{json.dumps(summary, indent=2, default=str)}\n```\n\n")
        for title, t in [("Table 3. Cross-validated comparison", t3), ("Table 4. Ablation", t4),
                         ("Table 5. Leakage check", t5), ("Table 6. Significance tests", pd.DataFrame(tests)),
                         ("Table 7. Manual-review band", pd.DataFrame(band)),
                         ("Holdout confusion matrix", holdout),
                         ("Permutation importance (top 15)", imp.head(15))]:
            f.write(f"## {title}\n\n{t.to_markdown(index=False, floatfmt='.4f')}\n\n")
    print(f"Done. See {a.out}/tables.md")


if __name__ == "__main__":
    main()
