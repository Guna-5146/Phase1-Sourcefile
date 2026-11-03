"""
Fidelity of the synthetic land data against real sources (Table 4 of the revised manuscript).

  1. Kolmogorov-Smirnov D and normalised Wasserstein distance per shared column
     (optionally per group, e.g. per village, reporting the median)
  2. Mean absolute difference between synthetic and real correlation matrices
  3. TSTR: train on synthetic, test on real, for a regression target (e.g. price),
     compared with train-on-real/test-on-real
  4. Chi-squared test of synthetic fraud-type counts against audit counts

Usage:
  python fidelity_check.py --synthetic land_transactions.csv --real guideline_values.csv \
      --columns market_value_per_acre --group village
  python fidelity_check.py --synthetic land_transactions.csv --real real_sales.csv \
      --columns price area --tstr-target price
  python fidelity_check.py --synthetic land_transactions.csv --fraud-type-col fraud_type \
      --audit-counts undervaluation=151 misclassification=47
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SEED = 42


def ks_wasserstein(syn, real, columns, group=None):
    rows = []
    for c in columns:
        if group and group in syn.columns and group in real.columns:
            per = []
            for g in sorted(set(syn[group]) & set(real[group])):
                s, r = syn.loc[syn[group] == g, c].dropna(), real.loc[real[group] == g, c].dropna()
                if len(s) > 1 and len(r) > 1:
                    per.append((stats.ks_2samp(s, r).statistic,
                                stats.wasserstein_distance(s, r) / (r.std() or 1)))
            d, w = np.median([p[0] for p in per]), np.median([p[1] for p in per])
            rows.append({"column": c, "KS D (median over groups)": d, "Wasserstein / SD (median)": w,
                         "groups": len(per)})
        else:
            s, r = syn[c].dropna(), real[c].dropna()
            rows.append({"column": c, "KS D": stats.ks_2samp(s, r).statistic,
                         "Wasserstein / SD": stats.wasserstein_distance(s, r) / (r.std() or 1)})
    return pd.DataFrame(rows)


def correlation_gap(syn, real, columns):
    cs, cr = syn[columns].corr().values, real[columns].corr().values
    return float(np.nanmean(np.abs(cs - cr)))


def tstr(syn, real, features, target):
    def fit_score(train, test):
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(train[features], np.log1p(train[target]))
        return r2_score(np.log1p(test[target]), m.predict(test[features]))
    real_tr, real_te = train_test_split(real.dropna(subset=features + [target]), test_size=0.3, random_state=SEED)
    return fit_score(syn.dropna(subset=features + [target]), real_te), fit_score(real_tr, real_te)


def fraud_mix(syn, col, audit):
    obs = syn[col].value_counts()
    kinds = list(audit)
    o = np.array([obs.get(k, 0) for k in kinds], dtype=float)
    e = np.array([audit[k] for k in kinds], dtype=float)
    e = e / e.sum() * o.sum()
    chi2, p = stats.chisquare(o, e)
    return pd.DataFrame({"type": kinds, "synthetic": o, "expected from audit": e.round(1)}), chi2, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--real")
    ap.add_argument("--columns", nargs="*", default=[])
    ap.add_argument("--group")
    ap.add_argument("--tstr-target")
    ap.add_argument("--fraud-type-col")
    ap.add_argument("--audit-counts", nargs="*", default=[], help="type=count pairs")
    a = ap.parse_args()

    syn = pd.read_csv(a.synthetic)
    if a.real and a.columns:
        real = pd.read_csv(a.real)
        print(ks_wasserstein(syn, real, a.columns, a.group).to_string(index=False), "\n")
        if len(a.columns) > 1:
            print(f"Mean absolute correlation difference: {correlation_gap(syn, real, a.columns):.4f}\n")
        if a.tstr_target:
            feats = [c for c in a.columns if c != a.tstr_target]
            r2_syn, r2_real = tstr(syn, real, feats, a.tstr_target)
            print(f"TSTR R2: {r2_syn:.4f}   (train-real/test-real R2: {r2_real:.4f})\n")
    if a.fraud_type_col and a.audit_counts:
        audit = {k: float(v) for k, v in (x.split("=") for x in a.audit_counts)}
        table, chi2, p = fraud_mix(syn[syn[a.fraud_type_col].notna()], a.fraud_type_col, audit)
        print(table.to_string(index=False))
        print(f"Chi-squared = {chi2:.3f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
