"""
Synthetic land-transaction generator for the revised manuscript (Section 3.2, Table 2).

Valuation comes from real guideline values (gv_village_summary.csv, produced by
collect_guideline_values.py). Fraud scenarios follow CAG audit categories and court
cases; the undervaluation : misclassification ratio follows the CAG Telangana counts
(151 : 47). Every rule and parameter below is printed to generator_report.md so it can
be copied into Table 2.

Usage:
  pip install numpy pandas
  python generate_land_data.py --gv gv/gv_village_summary.csv --n 80000 --out land_transactions.csv
  python generate_land_data.py --gv gv/gv_village_summary.csv --taluks Madurai-North Melur ...

Output: 34 columns (id, 31 attributes, fraud_type, is_fraud).
When training, drop the identifier and generator-only columns:
  --drop transaction_id seller_id buyer_id survey_number fraud_type
"""
import argparse

import numpy as np
import pandas as pd

SEED = 42

# Share of each scenario among fraudulent transactions (%). Undervaluation : misclassification
# = 45 : 14 = 3.2, matching 151 : 47 in CAG (2020). The other three come from court cases.
SHARES = {"undervaluation": 45, "misclassification": 14, "duplicate_sale": 15,
          "forged_poa": 14, "flipping": 12}

PARAMS = {
    "legit_price_ratio": "1 + |N(0.05, 0.15)|, with 4% of legitimate sales at U(0.85, 1.0)",
    "underval_ratio": (0.40, 0.80),       # declared price = U(a, b) x guideline value
    "dup_window_days": 90,                # same survey number, same seller, new buyer within T days
    "flip_window_days": 180,              # resold within T days
    "flip_multiplier": (1.8, 3.5),        # at k x previous price
    "stamp_duty_sale": 0.07,              # Tamil Nadu stamp duty on conveyance
    "stamp_duty_low": 0.01,               # settlement / gift to family
    "label_noise": 0.01,
}


def sample_villages(gv, n, rng):
    w = gv["surveys"] / gv["surveys"].sum()
    return gv.iloc[rng.choice(len(gv), size=n, p=w.values)].reset_index(drop=True)


def base_records(gv, n, rng, start="2019-01-01", end="2024-12-31"):
    v = sample_villages(gv, n, rng)
    agri = v["land_use"].eq("agricultural").values
    area = np.where(agri, rng.lognormal(np.log(0.9), 0.8, n), rng.uniform(0.03, 0.25, n))
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    date = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    ratio = 1 + np.abs(rng.normal(0.05, 0.15, n))
    low = rng.random(n) < 0.04
    ratio[low] = rng.uniform(0.85, 1.0, low.sum())       # honest sales near or below guideline
    guideline = area * v["market_value_per_acre"].values
    prev_days = rng.integers(200, 6000, n)
    df = pd.DataFrame({
        "transaction_id": np.arange(n),
        "transaction_date": date,
        "district": "Madurai",
        "taluk": v["taluk"].values,
        "sro": v["sro"].values,
        "village": v["village"].values,
        "survey_number": [f"{s}/{rng.integers(1, 9)}" for s in rng.integers(1, 900, n)],
        "land_use": v["land_use"].values,
        "area_acres": area.round(3),
        "market_value_per_acre": v["market_value_per_acre"].values.round(0),
        "price_inr": (guideline * ratio).round(-2),
        "document_type": rng.choice(["sale", "sale", "sale", "sale", "settlement", "gift", "partition"], n),
        "consideration_mode": rng.choice(["bank", "bank", "mixed", "cash"], n),
        "seller_id": rng.integers(1, n // 2, n),
        "buyer_id": rng.integers(n // 2, n * 2, n),
        "seller_txn_count": rng.poisson(0.6, n),
        "buyer_txn_count": rng.poisson(0.5, n),
        "ownership_years": (prev_days / 365).round(2),
        "days_since_prev_sale": prev_days,
        "prev_sale_price": (guideline * rng.uniform(0.5, 1.0, n)).round(-2),
        "poa_used": (rng.random(n) < 0.06).astype(int),
        "poa_holder_is_relative": 0,
        "seller_in_revenue_records": (rng.random(n) < 0.97).astype(int),
        "encumbrance_present": (rng.random(n) < 0.12).astype(int),
        "ec_checked": (rng.random(n) < 0.9).astype(int),
        "seller_age": rng.integers(25, 85, n),
        "buyer_age": rng.integers(22, 75, n),
        "witness_count": rng.choice([2, 2, 2, 3], n),
        "registration_lag_days": rng.poisson(4, n),
        "buyer_is_relative": 0,
    })
    fam = df["document_type"].isin(["settlement", "gift", "partition"])
    df.loc[fam, "buyer_is_relative"] = 1
    df.loc[fam, "consideration_mode"] = "none"
    df.loc[df["poa_used"] == 1, "poa_holder_is_relative"] = (rng.random((df["poa_used"] == 1).sum()) < 0.8).astype(int)
    df["stamp_duty_paid"] = np.where(fam, PARAMS["stamp_duty_low"], PARAMS["stamp_duty_sale"]) * \
        np.maximum(df["price_inr"], df["area_acres"] * df["market_value_per_acre"])
    df["fraud_type"] = "none"
    return df


def inject_fraud(df, n_fraud, rng):
    counts = {k: int(round(n_fraud * s / 100)) for k, s in SHARES.items()}
    idx = rng.permutation(df.index[df["fraud_type"] == "none"])
    pos = 0
    take = lambda k: idx[pos:pos + k]
    gv = lambda r: r["area_acres"] * r["market_value_per_acre"]

    # Undervaluation: declared price well below guideline value, duty paid on declared price
    i = take(counts["undervaluation"]); pos += len(i)
    a, b = PARAMS["underval_ratio"]
    df.loc[i, "price_inr"] = (gv(df.loc[i]) * rng.uniform(a, b, len(i))).round(-2)
    df.loc[i, "stamp_duty_paid"] = PARAMS["stamp_duty_sale"] * df.loc[i, "price_inr"]
    df.loc[i, "consideration_mode"] = rng.choice(["cash", "mixed"], len(i))
    df.loc[i, "fraud_type"] = "undervaluation"

    # Misclassification: a paid conveyance registered as a lower-duty instrument
    i = take(counts["misclassification"]); pos += len(i)
    df.loc[i, "document_type"] = rng.choice(["settlement", "gift"], len(i))
    df.loc[i, "consideration_mode"] = rng.choice(["bank", "cash"], len(i))
    df.loc[i, "buyer_is_relative"] = 0
    df.loc[i, "stamp_duty_paid"] = PARAMS["stamp_duty_low"] * df.loc[i, "price_inr"]
    df.loc[i, "fraud_type"] = "misclassification"

    # Forged power of attorney: PoA sale by a non-relative holder, seller absent from revenue records
    i = take(counts["forged_poa"]); pos += len(i)
    df.loc[i, "poa_used"] = 1
    df.loc[i, "poa_holder_is_relative"] = 0
    df.loc[i, "seller_in_revenue_records"] = (rng.random(len(i)) < 0.25).astype(int)
    df.loc[i, "ownership_years"] = rng.uniform(0, 1.5, len(i)).round(2)
    df.loc[i, "ec_checked"] = (rng.random(len(i)) < 0.5).astype(int)
    df.loc[i, "fraud_type"] = "forged_poa"

    # Flipping: resold within T days at k x previous price
    i = take(counts["flipping"]); pos += len(i)
    d = rng.integers(10, PARAMS["flip_window_days"], len(i))
    df.loc[i, "days_since_prev_sale"] = d
    df.loc[i, "ownership_years"] = (d / 365).round(2)
    df.loc[i, "price_inr"] = (gv(df.loc[i]) * rng.uniform(1.1, 1.6, len(i))).round(-2)
    lo, hi = PARAMS["flip_multiplier"]
    df.loc[i, "prev_sale_price"] = (df.loc[i, "price_inr"] / rng.uniform(lo, hi, len(i))).round(-2)
    df.loc[i, "seller_txn_count"] += rng.integers(2, 6, len(i))
    df.loc[i, "fraud_type"] = "flipping"

    # Duplicate sale: the same survey number sold again by the same seller to a new buyer within T days
    src = take(counts["duplicate_sale"]); pos += len(src)
    dup = df.loc[src].copy()
    dup["transaction_id"] = np.arange(df["transaction_id"].max() + 1, df["transaction_id"].max() + 1 + len(dup))
    dup["transaction_date"] = dup["transaction_date"] + pd.to_timedelta(
        rng.integers(1, PARAMS["dup_window_days"], len(dup)), unit="D")
    dup["buyer_id"] = rng.integers(len(df), len(df) * 2, len(dup))
    dup["ec_checked"] = (rng.random(len(dup)) < 0.3).astype(int)
    dup["fraud_type"] = "duplicate_sale"
    df = df.drop(index=idx[pos:pos + len(dup)])          # keep the total at n
    return pd.concat([df, dup], ignore_index=True)


def hard_negatives(df, rng, share=0.03):
    """Legitimate transactions that resemble fraud, so no single rule separates the classes."""
    legit = df.index[df["fraud_type"] == "none"]
    i = rng.choice(legit, int(share * len(legit)), replace=False)
    parts = np.array_split(i, 3)
    df.loc[parts[0], "poa_used"] = 1                                        # PoA held by a relative
    df.loc[parts[0], "poa_holder_is_relative"] = 1
    d = rng.integers(30, 365, len(parts[1]))                                # quick legitimate resale
    df.loc[parts[1], "days_since_prev_sale"] = d
    df.loc[parts[1], "ownership_years"] = (d / 365).round(2)
    df.loc[parts[1], "prev_sale_price"] = (df.loc[parts[1], "price_inr"] / rng.uniform(1.0, 1.6, len(d))).round(-2)
    df.loc[parts[2], "consideration_mode"] = "cash"                         # cash sale at fair price
    return df


def add_derived(df):
    df = df.sort_values("transaction_date").reset_index(drop=True)
    key = df["village"] + "|" + df["survey_number"]
    t = df["transaction_date"]
    cnt = np.zeros(len(df), dtype=int)
    for _, g in df.groupby(key).groups.items():
        g = list(g)
        for a_, b_ in zip(g, g[1:]):
            if (t[b_] - t[a_]).days <= PARAMS["dup_window_days"]:
                cnt[b_] += 1
    df["same_survey_sales_90d"] = cnt
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gv", required=True, help="gv_village_summary.csv")
    ap.add_argument("--taluks", nargs="*", help="keep only these taluks (Madurai district)")
    ap.add_argument("--land-use", nargs="*", default=["agricultural", "residential"])
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--fraud-rate", type=float, default=0.06)
    ap.add_argument("--out", default="land_transactions.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    gv = pd.read_csv(a.gv)
    gv = gv[gv["land_use"].isin(a.land_use) & (gv["market_value_per_acre"] > 0)]
    if a.taluks:
        gv = gv[gv["taluk"].str.lower().isin([t.lower() for t in a.taluks])]
    if gv.empty:
        raise SystemExit("No guideline values left after filtering.")

    df = base_records(gv, a.n, rng)
    df = inject_fraud(df, int(a.n * a.fraud_rate), rng)
    df = hard_negatives(df, rng)
    df["is_fraud"] = (df["fraud_type"] != "none").astype(int)
    flip = rng.random(len(df)) < PARAMS["label_noise"]
    df.loc[flip, "is_fraud"] = 1 - df.loc[flip, "is_fraud"]
    df = add_derived(df)
    cols = [c for c in df.columns if c not in ("fraud_type", "is_fraud")] + ["fraud_type", "is_fraud"]
    df = df[cols]
    df.to_csv(a.out, index=False)

    fr = df[df["fraud_type"] != "none"]["fraud_type"].value_counts(normalize=True).mul(100).round(1)
    with open(a.out.rsplit(".", 1)[0] + "_report.md", "w") as f:
        f.write(f"# Generator report\n\nRecords: {len(df):,}  Columns: {df.shape[1]}\n")
        f.write(f"Villages: {df['village'].nunique()}  Taluks: {df['taluk'].nunique()}\n")
        f.write(f"Fraudulent (is_fraud = 1): {df['is_fraud'].sum():,} ({100 * df['is_fraud'].mean():.1f}%)\n")
        f.write(f"Label noise: {100 * PARAMS['label_noise']:.0f}%  Hard negatives: 3% of legitimate\n\n")
        f.write("## Table 2 shares (% of generated fraud)\n\n" + fr.to_string() + "\n\n")
        f.write("## Table 2 rules\n\n")
        f.write(f"- Undervaluation: declared price = U{PARAMS['underval_ratio']} x guideline value\n")
        f.write("- Misclassification: paid sale (bank/cash consideration, non-relative buyer) registered as settlement or gift, duty 1% instead of 7%\n")
        f.write(f"- Duplicate sale: same survey number and seller sold to a new buyer within {PARAMS['dup_window_days']} days\n")
        f.write("- Forged PoA: PoA sale by a non-relative holder; seller absent from revenue records in 75% of cases\n")
        f.write(f"- Flipping: resold within {PARAMS['flip_window_days']} days at {PARAMS['flip_multiplier']} x previous price\n")
        f.write(f"- Legitimate price ratio: {PARAMS['legit_price_ratio']}\n")
    print(open(a.out.rsplit(".", 1)[0] + "_report.md").read())


if __name__ == "__main__":
    main()
